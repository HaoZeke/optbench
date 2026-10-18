/* File bridge from eOn ExtPot onto the optbench potential server.
   One process is one force call. The server owns the harness count. */
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static void die(const char *msg) {
    fprintf(stderr, "extpot-bridge: %s\n", msg);
    exit(1);
}

static int connect_endpoint(const char *endpoint) {
    char host[256];
    const char *colon = strrchr(endpoint, ':');
    if (!colon || (size_t)(colon - endpoint) >= sizeof(host)) die("bad endpoint");
    memcpy(host, endpoint, (size_t)(colon - endpoint));
    host[colon - endpoint] = 0;
    int port = atoi(colon + 1);
    if (port <= 0) die("bad port");
    struct addrinfo hints;
    struct addrinfo *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char port_text[16];
    snprintf(port_text, sizeof(port_text), "%d", port);
    if (getaddrinfo(host, port_text, &hints, &res) != 0) die("resolve");
    int fd = -1;
    for (struct addrinfo *it = res; it; it = it->ai_next) {
        fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        struct timeval tv;
        tv.tv_sec = 120;
        tv.tv_usec = 0;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (connect(fd, it->ai_addr, it->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) die("connect");
    return fd;
}

static void send_all(int fd, const char *buf, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t wrote = send(fd, buf + sent, n - sent, 0);
        if (wrote <= 0) die("send");
        sent += (size_t)wrote;
    }
}

int main(void) {
    const char *endpoint = getenv("OPTBENCH_POTSERVER");
    if (!endpoint || !strchr(endpoint, ':')) die("OPTBENCH_POTSERVER");
    FILE *in = fopen("from_eon_to_extpot", "r");
    if (!in) die("from_eon_to_extpot");
    char *lines[4096];
    int nlines = 0;
    char buf[512];
    while (fgets(buf, sizeof(buf), in)) {
        size_t len = strlen(buf);
        while (len && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) buf[--len] = 0;
        if (!len) continue;
        if (nlines >= 4096) die("too many atoms");
        lines[nlines] = strdup(buf);
        if (!lines[nlines]) die("strdup");
        nlines++;
    }
    fclose(in);
    if (nlines < 4) die("short request");
    int atoms = nlines - 3;
    char request[256 * 1024];
    int used = snprintf(request, sizeof(request), "calculate\n%d\n", atoms);
    for (int i = 0; i < 3; i++) {
        double a, b, c;
        if (sscanf(lines[i], "%lf %lf %lf", &a, &b, &c) != 3) die("box");
        used += snprintf(request + used, sizeof(request) - (size_t)used, "%.17g %.17g %.17g%s",
                         a, b, c, i == 2 ? "\n" : " ");
    }
    for (int i = 0; i < atoms; i++) {
        int z;
        double x, y, zc;
        if (sscanf(lines[3 + i], "%d %lf %lf %lf", &z, &x, &y, &zc) != 4) die("atom");
        used += snprintf(request + used, sizeof(request) - (size_t)used, "%d %.17g %.17g %.17g\n", z, x, y, zc);
        if (used >= (int)sizeof(request)) die("request overflow");
    }
    int fd = connect_endpoint(endpoint);
    send_all(fd, request, (size_t)used);
    shutdown(fd, SHUT_WR);
    char reply[256 * 1024];
    size_t got = 0;
    while (got + 1 < sizeof(reply)) {
        ssize_t n = recv(fd, reply + got, sizeof(reply) - 1 - got, 0);
        if (n == 0) break;
        if (n < 0) die("recv");
        got += (size_t)n;
    }
    close(fd);
    reply[got] = 0;
    if (strncmp(reply, "ok\n", 3) != 0) {
        fprintf(stderr, "extpot-bridge: %s\n", reply);
        /* A refused cap call must not leave the previous forces for the
           client to keep iterating. system() ignores our status, so stop
           the client. The shell is our parent; the client is its parent. */
        unlink("from_extpot_to_eon");
        /* Only a harness cap is fatal. Other server errors still drop the
           stale force file; the client then has nothing to read. */
        if (!strstr(reply, "cap")) return 1;
        pid_t shell = getppid();
        pid_t client = shell;
        char path[64];
        snprintf(path, sizeof(path), "/proc/%d/status", (int)shell);
        FILE *status = fopen(path, "r");
        if (status) {
            char line[256];
            while (fgets(line, sizeof(line), status)) {
                int ppid = 0;
                if (sscanf(line, "PPid: %d", &ppid) == 1) {
                    client = (pid_t)ppid;
                    break;
                }
            }
            fclose(status);
        }
        if (client > 1 && client != shell) kill(client, SIGKILL);
        if (shell > 1) kill(shell, SIGKILL);
        return 1;
    }
    char *save = NULL;
    char *line = strtok_r(reply, "\n", &save);
    line = strtok_r(NULL, "\n", &save);
    line = strtok_r(NULL, "\n", &save);
    line = strtok_r(NULL, "\n", &save);
    if (!line) die("short reply");
    FILE *out = fopen("from_extpot_to_eon", "w");
    if (!out) die("from_extpot_to_eon");
    fprintf(out, "%s\n", line);
    for (int i = 0; i < atoms; i++) {
        line = strtok_r(NULL, "\n", &save);
        if (!line) die("short forces");
        fprintf(out, "%s\n", line);
    }
    fclose(out);
    for (int i = 0; i < nlines; i++) free(lines[i]);
    return 0;
}
