/* Stop OPTIM when the next Morse gradient call would pass the cap.
 *
 * morse_bulk_wrapper_ is the native energy evaluation. Its fourth
 * argument is the logical GTEST. OPTIM counts a call in the
 * energy-plus-gradient total only when that logical is true. The
 * breakpoint is the first instruction, so the call is refused before
 * it runs.
 */
#define _GNU_SOURCE

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

static void die(const char *what) {
    fprintf(stderr, "optim_callcap: %s: %s\n", what, strerror(errno));
    exit(1);
}

static long peek(pid_t pid, unsigned long addr) {
    errno = 0;
    long word = ptrace(PTRACE_PEEKDATA, pid, (void *)addr, 0);
    if (word == -1 && errno != 0) {
        die("peek");
    }
    return word;
}

static void poke_text(pid_t pid, unsigned long addr, unsigned long word) {
    if (ptrace(PTRACE_POKETEXT, pid, (void *)addr, (void *)word) < 0) {
        die("poketext");
    }
}

static int read_i32(pid_t pid, unsigned long addr) {
    unsigned long aligned = addr & ~7UL;
    unsigned shift = (unsigned)(addr & 7UL) * 8U;
    unsigned long word = (unsigned long)peek(pid, aligned);
    return (int)((word >> shift) & 0xffffffffu);
}

static unsigned long load_base(pid_t pid, const char *binary) {
    char path[64];
    char line[1024];
    FILE *handle;
    snprintf(path, sizeof path, "/proc/%d/maps", (int)pid);
    handle = fopen(path, "r");
    if (handle == NULL) {
        die("maps");
    }
    while (fgets(line, sizeof line, handle) != NULL) {
        unsigned long start = 0;
        unsigned long end = 0;
        unsigned long offset = 0;
        char perms[8];
        if (sscanf(line, "%lx-%lx %7s %lx", &start, &end, perms, &offset) != 4) {
            continue;
        }
        if (perms[0] != 'r' || perms[2] != 'x') {
            continue;
        }
        if (strstr(line, binary) == NULL) {
            continue;
        }
        fclose(handle);
        return start - offset;
    }
    fclose(handle);
    fprintf(stderr, "optim_callcap: %s is not mapped\n", binary);
    exit(1);
}

static void write_status(const char *path, long gradient, long other, int capped, int child) {
    char tmp[768];
    FILE *handle;
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    handle = fopen(tmp, "w");
    if (handle == NULL) {
        die("status");
    }
    fprintf(handle, "gradient %ld\nother %ld\ncapped %d\nchild %d\n", gradient, other, capped, child);
    if (fclose(handle) != 0) {
        die("status");
    }
    if (rename(tmp, path) != 0) {
        die("status");
    }
}

int main(int argc, char **argv) {
    unsigned long symbol;
    long cap;
    const char *status_path;
    const char *binary;
    pid_t pid;
    int status;
    unsigned long base;
    unsigned long breakpoint;
    unsigned long word_addr;
    unsigned shift;
    unsigned long saved;
    long gradient = 0;
    long other = 0;
    int capped = 0;
    int stepping = 0;
    char *child_argv[2];

    if (argc != 5) {
        fprintf(stderr, "usage: optim_callcap <symbol_vma> <cap> <status> <binary>\n");
        return 2;
    }
    symbol = strtoul(argv[1], NULL, 0);
    cap = strtol(argv[2], NULL, 10);
    status_path = argv[3];
    binary = argv[4];
    if (cap < 1) {
        fprintf(stderr, "optim_callcap: cap must be positive\n");
        return 2;
    }

    pid = fork();
    if (pid < 0) {
        die("fork");
    }
    if (pid == 0) {
        if (ptrace(PTRACE_TRACEME, 0, 0, 0) < 0) {
            _exit(126);
        }
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (getppid() == 1) {
            _exit(125);
        }
        child_argv[0] = (char *)binary;
        child_argv[1] = NULL;
        execv(binary, child_argv);
        _exit(127);
    }

    if (waitpid(pid, &status, 0) < 0) {
        die("wait");
    }
    if (!WIFSTOPPED(status)) {
        fprintf(stderr, "optim_callcap: OPTIM exited before tracing started\n");
        return 1;
    }
    if (ptrace(PTRACE_SETOPTIONS, pid, 0, (void *)(uintptr_t)PTRACE_O_EXITKILL) < 0) {
        die("setoptions");
    }
    base = load_base(pid, binary);
    breakpoint = base + symbol;
    word_addr = breakpoint & ~7UL;
    shift = (unsigned)(breakpoint & 7UL) * 8U;
    saved = (unsigned long)peek(pid, word_addr);
    poke_text(pid, word_addr, (saved & ~(0xffUL << shift)) | (0xccUL << shift));
    if (ptrace(PTRACE_CONT, pid, 0, 0) < 0) {
        die("cont");
    }

    for (;;) {
        if (waitpid(pid, &status, 0) < 0) {
            if (errno == EINTR) {
                continue;
            }
            die("wait");
        }
        if (WIFEXITED(status) || WIFSIGNALED(status)) {
            int child = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
            write_status(status_path, gradient, other, capped, child);
            return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
        }
        if (!WIFSTOPPED(status)) {
            continue;
        }
        if (stepping) {
            if (WSTOPSIG(status) != SIGTRAP) {
                if (ptrace(PTRACE_SINGLESTEP, pid, 0, (void *)(uintptr_t)WSTOPSIG(status)) < 0) {
                    die("singlestep");
                }
                continue;
            }
            unsigned long patched = (saved & ~(0xffUL << shift)) | (0xccUL << shift);
            poke_text(pid, word_addr, patched);
            stepping = 0;
            if (ptrace(PTRACE_CONT, pid, 0, 0) < 0) {
                die("cont");
            }
            continue;
        }
        if (WSTOPSIG(status) != SIGTRAP) {
            if (ptrace(PTRACE_CONT, pid, 0, (void *)(uintptr_t)WSTOPSIG(status)) < 0) {
                die("cont");
            }
            continue;
        }
        {
            struct user_regs_struct regs;
            int gtest;
            if (ptrace(PTRACE_GETREGS, pid, 0, &regs) < 0) {
                die("getregs");
            }
            if (regs.rip != breakpoint + 1 && regs.rip != breakpoint) {
                if (ptrace(PTRACE_CONT, pid, 0, 0) < 0) {
                    die("cont");
                }
                continue;
            }
            gtest = read_i32(pid, (unsigned long)regs.rcx);
            if (gtest) {
                if (gradient >= cap) {
                    capped = 1;
                    write_status(status_path, gradient, other, capped, 0);
                    kill(pid, SIGKILL);
                    waitpid(pid, &status, 0);
                    return 0;
                }
                gradient++;
            } else {
                other++;
            }
            regs.rip = breakpoint;
            if (ptrace(PTRACE_SETREGS, pid, 0, &regs) < 0) {
                die("setregs");
            }
            poke_text(pid, word_addr, saved);
            stepping = 1;
            if (ptrace(PTRACE_SINGLESTEP, pid, 0, 0) < 0) {
                die("singlestep");
            }
        }
    }
}
