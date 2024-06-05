#include <stdio.h>
#include <stdlib.h>

void read_memory(void *address, size_t size) {
    unsigned char *ptr = (unsigned char *)address;
    printf("Reading memory at address: %p\n", address);
    for (size_t i = 0; i < size; i++) {
        printf("%02x ", ptr[i]);
    }
    printf("\n");
}

int main() {
    // This main function is not used when calling from Python
    return 0;
}