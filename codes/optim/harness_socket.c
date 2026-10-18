/* OptBench potential client for OPTIM.

   One POTENTIAL entry is one TCP calculate. The server counts that
   request. A ForceCallCap reply is status 2. Any other failure is
   status 1. Status 0 fills energy and the Cartesian gradient.
*/
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int read_line(int fd, char *buf, int buflen) {
    int used = 0;
    while (used + 1 < buflen) {
        char byte = 0;
        ssize_t got = recv(fd, &byte, 1, 0);
        if (got == 0) {
            buf[used] = '\0';
            return used == 0 ? -1 : 0;
        }
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (byte == '\n') {
            buf[used] = '\0';
            return 0;
        }
        if (byte != '\r') {
            buf[used++] = byte;
        }
    }
    buf[buflen - 1] = '\0';
    return -1;
}

static int connect_endpoint(const char *endpoint) {
    char host[256];
    char *colon;
    char *end = NULL;
    long port;
    int fd;
    struct sockaddr_in address;
    int flag = 1;

    if (strlen(endpoint) >= sizeof(host)) {
        return -1;
    }
    memcpy(host, endpoint, strlen(endpoint) + 1);
    colon = strrchr(host, ':');
    if (colon == NULL || colon == host) {
        return -1;
    }
    *colon = '\0';
    port = strtol(colon + 1, &end, 10);
    if (end == colon + 1 || *end != '\0' || port < 1 || port > 65535) {
        return -1;
    }
    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host, &address.sin_addr) != 1) {
        struct addrinfo hints;
        struct addrinfo *found = NULL;
        char port_text[16];
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        snprintf(port_text, sizeof(port_text), "%ld", port);
        if (getaddrinfo(host, port_text, &hints, &found) != 0 || found == NULL) {
            close(fd);
            return -1;
        }
        if (connect(fd, found->ai_addr, found->ai_addrlen) != 0) {
            freeaddrinfo(found);
            close(fd);
            return -1;
        }
        freeaddrinfo(found);
        return fd;
    }
    if (connect(fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int harness_eval(int natoms, const double *coords, double *energy, double *grad) {
    const char *endpoint = getenv("OPTBENCH_POTSERVER");
    int fd;
    int atom;
    char line[512];
    char *cursor;
    int status;

    if (endpoint == NULL || endpoint[0] == '\0' || natoms < 1 || coords == NULL || energy == NULL || grad == NULL) {
        return 1;
    }
    fd = connect_endpoint(endpoint);
    if (fd < 0) {
        return 1;
    }
    if (send(fd, "calculate\n", 10, MSG_NOSIGNAL) != 10) {
        close(fd);
        return 1;
    }
    snprintf(line, sizeof(line), "%d\n", natoms);
    if (send(fd, line, strlen(line), MSG_NOSIGNAL) < 0) {
        close(fd);
        return 1;
    }
    if (send(fd, "0 0 0 0 0 0 0 0 0\n", 18, MSG_NOSIGNAL) != 18) {
        close(fd);
        return 1;
    }
    for (atom = 0; atom < natoms; atom++) {
        int wrote = snprintf(
            line,
            sizeof(line),
            "1 %.17g %.17g %.17g\n",
            coords[3 * atom],
            coords[3 * atom + 1],
            coords[3 * atom + 2]
        );
        if (wrote < 0 || wrote >= (int)sizeof(line) || send(fd, line, (size_t)wrote, MSG_NOSIGNAL) != wrote) {
            close(fd);
            return 1;
        }
    }
    if (read_line(fd, line, sizeof(line)) != 0) {
        close(fd);
        return 1;
    }
    if (strcmp(line, "ok") != 0) {
        status = 1;
        if (strstr(line, "ForceCallCap") != NULL) {
            status = 2;
        } else if (strcmp(line, "error") == 0 && read_line(fd, line, sizeof(line)) == 0 && strstr(line, "ForceCallCap") != NULL) {
            status = 2;
        }
        close(fd);
        return status;
    }
    if (read_line(fd, line, sizeof(line)) != 0 || read_line(fd, line, sizeof(line)) != 0 || read_line(fd, line, sizeof(line)) != 0) {
        close(fd);
        return 1;
    }
    *energy = strtod(line, &cursor);
    if (cursor == line) {
        close(fd);
        return 1;
    }
    for (atom = 0; atom < natoms; atom++) {
        char *next = NULL;
        if (read_line(fd, line, sizeof(line)) != 0) {
            close(fd);
            return 1;
        }
        grad[3 * atom] = strtod(line, &next);
        if (next == line) {
            close(fd);
            return 1;
        }
        grad[3 * atom + 1] = strtod(next, &cursor);
        if (cursor == next) {
            close(fd);
            return 1;
        }
        grad[3 * atom + 2] = strtod(cursor, &next);
        if (next == cursor) {
            close(fd);
            return 1;
        }
    }
    close(fd);
    return 0;
}
