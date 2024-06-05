#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    printf("running crazy.c --- Trying stupidity\n");

    volatile __uint64_t loc = 0x7f374856abc0;

    printf("location of my value: %lp\n\n", &loc);
    printf("in decimal form: %ld\n\n", &loc);
    int i = 0;
    while (i < 20) 
    {
        printf("my value is %lu\n", loc);
        sleep(1);
        i += 1;
    }

    volatile char *cheat = NULL;
    cheat = (char *) loc;

    // volatile void *out = mmap(cheat, 272, PROT_READ | PROT_EXEC | PROT_WRITE, MAP_SHARED, 0, 0);

    // cheat = (char *) out;
    // cheat[1] = 1;
}