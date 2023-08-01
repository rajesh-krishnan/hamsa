#ifndef _MMAP_HELPER_H
#define _MMAP_HELPER_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>
#include <limits.h>
#include <sys/mman.h>
#include <linux/mman.h>
#include <asm-generic/mman-common.h>

void *mymap (size_t size);
void myunmap (void *ptr, size_t size);

#endif /* _MMAP_HELPER_H */
