/* In-process ext_pot exchange.

   eonclient calls system() on the extpot-bridge path for every force
   evaluation. That forks a process and resolves the harness host each
   time. The dialogue itself is one short TCP request. Loaded with
   LD_PRELOAD, this library keeps that dialogue and skips the fork.
   The reply file and the HarnessCounter charge stay the same. Any other
   system() command is passed through.
*/
#define _GNU_SOURCE

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static pthread_mutex_t exchange_lock = PTHREAD_MUTEX_INITIALIZER;

static void die(const char *message) {
    fprintf(stderr, "optbench ext_pot: %s\n", message);
    exit(1);
}

static char *read_all(const char *path, size_t *length) {
    FILE *file = fopen(path, "rb");
    char *data;
    long size;
    if (file == NULL) {
        die("cannot read from_eon_to_extpot");
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        die("cannot size from_eon_to_extpot");
    }
    size = ftell(file);
    if (size < 0) {
        die("cannot size from_eon_to_extpot");
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        die("cannot rewind from_eon_to_extpot");
    }
    data = (char *)malloc((size_t)size + 1);
    if (data == NULL) {
        die("out of memory");
    }
    if (size > 0 && fread(data, 1, (size_t)size, file) != (size_t)size) {
        die("short read of from_eon_to_extpot");
    }
    data[size] = '\0';
    fclose(file);
    *length = (size_t)size;
    return data;
}

static int connect_server(const char *host, const char *port) {
    struct sockaddr_in address;
    int fd;
    int one = 1;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons((unsigned short)atoi(port));
    if (inet_pton(AF_INET, host, &address.sin_addr) != 1) {
        die("OPTBENCH_POTSERVER host is not an IPv4 address");
    }
    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0 || connect(fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        if (fd >= 0) {
            close(fd);
        }
        die("cannot connect to OPTBENCH_POTSERVER");
    }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

static void send_all(int fd, const char *data, size_t length) {
    size_t sent = 0;
    while (sent < length) {
        ssize_t wrote = send(fd, data + sent, length - sent, MSG_NOSIGNAL);
        if (wrote < 0) {
            if (errno == EINTR) {
                continue;
            }
            die("cannot write the calculate request");
        }
        sent += (size_t)wrote;
    }
}

static char *recv_all(int fd, size_t *length) {
    size_t used = 0;
    size_t cap = 4096;
    char *data = (char *)malloc(cap);
    if (data == NULL) {
        die("out of memory");
    }
    for (;;) {
        ssize_t got;
        if (used + 2048 > cap) {
            cap *= 2;
            data = (char *)realloc(data, cap);
            if (data == NULL) {
                die("out of memory");
            }
        }
        got = recv(fd, data + used, cap - used - 1, 0);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            die("cannot read the potential reply");
        }
        if (got == 0) {
            break;
        }
        used += (size_t)got;
    }
    data[used] = '\0';
    *length = used;
    return data;
}

static char *next_line(char **cursor) {
    char *start;
    char *end;
    if (cursor == NULL || *cursor == NULL) {
        return NULL;
    }
    start = *cursor;
    if (*start == '\0') {
        *cursor = NULL;
        return NULL;
    }
    end = strchr(start, '\n');
    if (end == NULL) {
        *cursor = NULL;
        return start;
    }
    *end = '\0';
    *cursor = end + 1;
    return start;
}

static void append(char **payload, size_t *used, size_t *cap, const char *text) {
    size_t length = strlen(text);
    if (*used + length + 1 >= *cap) {
        *cap = (*used + length + 64) * 2;
        *payload = (char *)realloc(*payload, *cap);
        if (*payload == NULL) {
            die("out of memory");
        }
    }
    memcpy(*payload + *used, text, length);
    *used += length;
    (*payload)[*used] = '\0';
}

/* develop's ext_pot runs `cd '<exchange dir>' && ... && extpot-bridge`.
   The request file is in that directory, not in the client cwd. r2549
   calls system() on the bridge path with the file already in cwd. */
static int exchange_directory(const char *command, char *directory, size_t cap) {
    const char *start;
    const char *end;
    size_t length;
    if (strncmp(command, "cd '", 4) != 0) {
        directory[0] = '\0';
        return 0;
    }
    start = command + 4;
    end = strchr(start, '\'');
    if (end == NULL) {
        return -1;
    }
    length = (size_t)(end - start);
    if (length + 1 >= cap) {
        return -1;
    }
    memcpy(directory, start, length);
    directory[length] = '\0';
    return 1;
}

static int exchange(const char *directory) {
    const char *endpoint = getenv("OPTBENCH_POTSERVER");
    const char *colon;
    char host[256];
    char port[32];
    char *request;
    char *reply;
    char *payload;
    char *cursor;
    char *line;
    char **lines = NULL;
    size_t request_length = 0;
    size_t reply_length = 0;
    size_t payload_used = 0;
    size_t payload_cap = 0;
    size_t line_count = 0;
    size_t line_cap = 0;
    size_t index;
    int atoms;
    int fd;
    int reply_line = 0;
    int force_rows = 0;
    int cap_refusal = 0;
    FILE *out;
    char count_line[64];
    char request_path[4096];
    char reply_path[4096];

    if (directory != NULL && directory[0] != '\0') {
        snprintf(request_path, sizeof(request_path), "%s/from_eon_to_extpot", directory);
        snprintf(reply_path, sizeof(reply_path), "%s/from_extpot_to_eon", directory);
    } else {
        snprintf(request_path, sizeof(request_path), "from_eon_to_extpot");
        snprintf(reply_path, sizeof(reply_path), "from_extpot_to_eon");
    }

    if (endpoint == NULL || (colon = strrchr(endpoint, ':')) == NULL) {
        die("OPTBENCH_POTSERVER must be host:port");
    }
    if ((size_t)(colon - endpoint) == 0 || (size_t)(colon - endpoint) >= sizeof(host)) {
        die("OPTBENCH_POTSERVER host is empty");
    }
    memcpy(host, endpoint, (size_t)(colon - endpoint));
    host[colon - endpoint] = '\0';
    if (strlen(colon + 1) == 0 || strlen(colon + 1) >= sizeof(port)) {
        die("OPTBENCH_POTSERVER port is empty");
    }
    memcpy(port, colon + 1, strlen(colon + 1) + 1);

    request = read_all(request_path, &request_length);
    cursor = request;
    while ((line = next_line(&cursor)) != NULL) {
        char *end = line + strlen(line);
        char *tab;
        while (end > line && (end[-1] == '\r' || end[-1] == ' ' || end[-1] == '\t')) {
            *--end = '\0';
        }
        if (line[0] == '\0') {
            continue;
        }
        for (tab = line; *tab != '\0'; tab++) {
            if (*tab == '\t') {
                *tab = ' ';
            }
        }
        if (line_count == line_cap) {
            line_cap = line_cap == 0 ? 64 : line_cap * 2;
            lines = (char **)realloc(lines, line_cap * sizeof(char *));
            if (lines == NULL) {
                die("out of memory");
            }
        }
        lines[line_count++] = line;
    }
    if (line_count < 4) {
        die("ext_pot request is shorter than a box plus one atom");
    }
    atoms = (int)line_count - 3;

    payload_cap = request_length + 64;
    payload = (char *)malloc(payload_cap);
    if (payload == NULL) {
        die("out of memory");
    }
    payload[0] = '\0';
    snprintf(count_line, sizeof(count_line), "calculate\n%d\n", atoms);
    append(&payload, &payload_used, &payload_cap, count_line);
    for (index = 0; index < 3; index++) {
        append(&payload, &payload_used, &payload_cap, lines[index]);
        append(&payload, &payload_used, &payload_cap, index + 1 < 3 ? " " : "\n");
    }
    for (index = 3; index < line_count; index++) {
        append(&payload, &payload_used, &payload_cap, lines[index]);
        append(&payload, &payload_used, &payload_cap, "\n");
    }
    free(lines);
    free(request);

    fd = connect_server(host, port);
    send_all(fd, payload, payload_used);
    shutdown(fd, SHUT_WR);
    free(payload);
    reply = recv_all(fd, &reply_length);
    close(fd);
    if (reply_length == 0) {
        die("potential server closed without a reply");
    }
    cap_refusal = strstr(reply, "cap") != NULL;

    out = fopen(reply_path, "w");
    if (out == NULL) {
        die("cannot write from_extpot_to_eon");
    }
    cursor = reply;
    while ((line = next_line(&cursor)) != NULL) {
        if (line[0] == '\0') {
            continue;
        }
        if (reply_line == 0) {
            if (strcmp(line, "ok") != 0) {
                fclose(out);
                unlink(reply_path);
                fprintf(stderr, "optbench ext_pot: %s\n", line);
                /* Inside the client, a harness cap has to stop this process.
                   The forked bridge killed its parent; that parent is us. */
                if (cap_refusal) {
                    free(reply);
                    kill(getpid(), SIGKILL);
                }
                free(reply);
                exit(1);
            }
        } else if (reply_line >= 3) {
            fprintf(out, "%s\n", line);
            if (reply_line > 3) {
                force_rows += 1;
            }
        }
        reply_line += 1;
    }
    free(reply);
    if (fclose(out) != 0) {
        die("cannot close from_extpot_to_eon");
    }
    if (force_rows != atoms) {
        die("potential reply has the wrong number of force rows");
    }
    return 0;
}

int system(const char *command) {
    static int (*real_system)(const char *) = NULL;
    int status;
    if (real_system == NULL) {
        real_system = (int (*)(const char *))dlsym(RTLD_NEXT, "system");
        if (real_system == NULL) {
            return -1;
        }
    }
    if (command != NULL && strstr(command, "extpot-bridge") != NULL && getenv("OPTBENCH_POTSERVER") != NULL) {
        char directory[4096];
        int located = exchange_directory(command, directory, sizeof(directory));
        if (located < 0) {
            return real_system(command);
        }
        pthread_mutex_lock(&exchange_lock);
        status = exchange(directory);
        pthread_mutex_unlock(&exchange_lock);
        return status;
    }
    return real_system(command);
}
