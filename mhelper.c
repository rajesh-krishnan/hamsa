#include "mhelper.h"
#include <stdio.h>
#include <stdlib.h>
#include <linux/mman.h>
#include <sys/mman.h>
#include <asm-generic/mman-common.h>

void *mymap (size_t size) {
    void *ptr;

    ptr = mmap(NULL, size, PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (ptr == MAP_FAILED){
        fprintf(stderr, "hugetlb mmap failed %ld\n", size);
        ptr = mmap(NULL, size, PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
    if (ptr == MAP_FAILED){
        fprintf(stderr, "mmap failed \n");
        exit(0);
    }
    return ptr;
}

void myunmap (void *ptr, size_t size) { munmap(ptr, size); }

