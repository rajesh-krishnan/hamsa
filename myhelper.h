#ifndef _MYHELPER_H
#define _MYHELPER_H
#include <omp.h>
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

#include "hdefs.h"
#include "mt19937/mt19937ar.h"

void *mymap (size_t size);
void myunmap (void *ptr, size_t size);
void myrnginit();
void myshuffle(int *array, int n);

#endif /* _MYHELPER_H */
