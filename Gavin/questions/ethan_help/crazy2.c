#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    volatile __uint64_t in = (__uint64_t) atol(argv[1]);

    volatile __uint64_t *p = (__uint64_t *) in;

    *p = 5;
}