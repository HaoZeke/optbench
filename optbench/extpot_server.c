/* Harness potential server for the Pt-island Morse kernel.

   The dialogue matches PotServer: one command per connection. The pair
   sum is the same shared library morse_smooth_linear loads, so the
   energy and forces are that kernel. Forces are the negative gradient.
   The island is periodic in x and y.
*/
#define _GNU_SOURCE

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

typedef double (*morse_fn)(
    const double *positions,
    double *gradient,
    int count,
    double depth,
    double alpha,
    double r0,
    double cutoff,
    double cell_x,
    double cell_y,
    double cell_z,
    int periodic_x,
    int periodic_y,
    int periodic_z
);

static morse_fn morse = NULL;
static long calls = 0;
static long verify_calls = 0;
static int verifying = 0;
static long cap = -1;

static void die(const char *message) {
    fprintf(stderr, "optbench extpot server: %s\n", message);
    exit(1);
}

typedef struct {
    int fd;
    char buf[65536];
    size_t start;
    size_t end;
} Reader;

static int read_line(Reader *reader, char *out, size_t cap_bytes) {
    size_t used = 0;
    for (;;) {
        while (reader->start < reader->end) {
            char ch = reader->buf[reader->start++];
            if (ch == '\n') {
                out[used] = '\0';
                return 1;
            }
            if (ch == '\r') {
                continue;
            }
            if (used + 1 < cap_bytes) {
                out[used++] = ch;
            }
        }
        ssize_t got = recv(reader->fd, reader->buf, sizeof(reader->buf), 0);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            return 0;
        }
        if (got == 0) {
            return 0;
        }
        reader->start = 0;
        reader->end = (size_t)got;
    }
}

static void send_text(int fd, const char *text) {
    size_t length = strlen(text);
    size_t sent = 0;
    while (sent < length) {
        ssize_t wrote = send(fd, text + sent, length - sent, MSG_NOSIGNAL);
        if (wrote < 0) {
            if (errno == EINTR) {
                continue;
            }
            return;
        }
        sent += (size_t)wrote;
    }
}

static int read_floats(const char *line, double *out, int count) {
    const char *cursor = line;
    int index;
    for (index = 0; index < count; index++) {
        char *end = NULL;
        out[index] = strtod(cursor, &end);
        if (end == cursor) {
            return -1;
        }
        cursor = end;
    }
    return 0;
}

static void reply_error(int fd, const char *message) {
    char text[256];
    snprintf(text, sizeof(text), "error\n%s\n", message);
    send_text(fd, text);
}

static void handle_calculate(int fd, Reader *reader, const char *count_line) {
    int atoms;
    int index;
    double box[9];
    double *positions;
    double *gradient;
    double energy;
    char *reply;
    size_t used = 0;
    size_t room;
    char *end = NULL;
    char line[512];
    atoms = (int)strtol(count_line, &end, 10);
    if (end == count_line || atoms < 1 || atoms > 100000) {
        reply_error(fd, "atom count must be positive");
        return;
    }
    if (!read_line(reader, line, sizeof(line)) || read_floats(line, box, 9) != 0) {
        reply_error(fd, "box is nine floats");
        return;
    }
    if (!verifying && cap >= 0 && calls >= cap) {
        char message[64];
        snprintf(message, sizeof(message), "ForceCallCap: force-call cap %ld", cap);
        reply_error(fd, message);
        return;
    }
    positions = (double *)malloc((size_t)atoms * 3u * sizeof(double));
    gradient = (double *)malloc((size_t)atoms * 3u * sizeof(double));
    if (positions == NULL || gradient == NULL) {
        free(positions);
        free(gradient);
        reply_error(fd, "out of memory");
        return;
    }
    for (index = 0; index < atoms; index++) {
        char *field = NULL;
        double coord[3];
        if (!read_line(reader, line, sizeof(line))) {
            free(positions);
            free(gradient);
            reply_error(fd, "calculate atom list is short");
            return;
        }
        strtol(line, &field, 10);
        if (field == NULL || read_floats(field, coord, 3) != 0) {
            free(positions);
            free(gradient);
            reply_error(fd, "an atom line is Z x y z");
            return;
        }
        positions[3 * index] = coord[0];
        positions[3 * index + 1] = coord[1];
        positions[3 * index + 2] = coord[2];
    }
    if (verifying) {
        verify_calls += 1;
    } else {
        calls += 1;
    }
    energy = morse(
        positions,
        gradient,
        atoms,
        0.7102,
        1.6047,
        2.897,
        9.5,
        box[0],
        box[4],
        box[8],
        1,
        1,
        0
    );
    room = (size_t)atoms * 96u + 128u;
    reply = (char *)malloc(room);
    if (reply == NULL) {
        free(positions);
        free(gradient);
        die("out of memory");
    }
    used += (size_t)snprintf(reply + used, room - used, "ok\n%ld\n%ld\n%.17g\n", calls, verify_calls, energy);
    for (index = 0; index < atoms; index++) {
        used += (size_t)snprintf(
            reply + used,
            room - used,
            "%.17g %.17g %.17g\n",
            -gradient[3 * index],
            -gradient[3 * index + 1],
            -gradient[3 * index + 2]
        );
    }
    send_text(fd, reply);
    free(reply);
    free(positions);
    free(gradient);
}

static void handle(int fd) {
    Reader reader;
    char line[512];
    memset(&reader, 0, sizeof(reader));
    reader.fd = fd;
    if (!read_line(&reader, line, sizeof(line))) {
        return;
    }
    if (strcmp(line, "count") == 0) {
        char text[128];
        snprintf(text, sizeof(text), "count\n%ld\n%ld\n", calls, verify_calls);
        send_text(fd, text);
    } else if (strcmp(line, "reset") == 0) {
        calls = 0;
        verify_calls = 0;
        verifying = 0;
        send_text(fd, "ok\n");
    } else if (strcmp(line, "verify") == 0) {
        char flag[32];
        if (!read_line(&reader, flag, sizeof(flag)) || (strcmp(flag, "on") != 0 && strcmp(flag, "off") != 0)) {
            reply_error(fd, "verify needs on or off");
        } else {
            verifying = strcmp(flag, "on") == 0;
            send_text(fd, "ok\n");
        }
    } else if (strcmp(line, "bye") == 0) {
        send_text(fd, "bye\n");
    } else if (strcmp(line, "calculate") == 0) {
        char count_line[64];
        if (!read_line(&reader, count_line, sizeof(count_line))) {
            reply_error(fd, "calculate needs a count and a box");
            return;
        }
        handle_calculate(fd, &reader, count_line);
    } else {
        reply_error(fd, "unknown command");
    }
}

int main(int argc, char **argv) {
    int listen_fd;
    struct sockaddr_in address;
    socklen_t length;
    int port;
    int option = 1;
    int index;
    const char *kernel = NULL;
    void *library;
    signal(SIGPIPE, SIG_IGN);
    for (index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--cap") == 0 && index + 1 < argc) {
            cap = strtol(argv[++index], NULL, 10);
        } else if (strcmp(argv[index], "--kernel") == 0 && index + 1 < argc) {
            kernel = argv[++index];
        }
    }
    if (kernel == NULL) {
        die("pass --kernel");
    }
    library = dlopen(kernel, RTLD_NOW);
    if (library == NULL) {
        fprintf(stderr, "optbench extpot server: %s\n", dlerror());
        return 1;
    }
    morse = (morse_fn)dlsym(library, "morse_pair_energy");
    if (morse == NULL) {
        die("morse_pair_energy is missing");
    }
    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        die("socket");
    }
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(listen_fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        die("bind");
    }
    if (listen(listen_fd, 16) != 0) {
        die("listen");
    }
    length = sizeof(address);
    if (getsockname(listen_fd, (struct sockaddr *)&address, &length) != 0) {
        die("getsockname");
    }
    port = ntohs(address.sin_port);
    printf("127.0.0.1:%d\n", port);
    fflush(stdout);
    for (;;) {
        int fd = accept(listen_fd, NULL, NULL);
        int one = 1;
        if (fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            die("accept");
        }
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        handle(fd);
        close(fd);
    }
    return 0;
}
