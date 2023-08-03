#pragma once
#include <omp.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
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

void *mymap(size_t size);
void myunmap(void *ptr, size_t size);
void myrnginit();
void myshuffle(int *array, int n);
float randnorm(double mu, double sigma);
void write_fnpy(float *farr, bool twoD, size_t d0, size_t d1, char *fn);
void read_fnpy(float *farr, bool twoD, size_t d0, size_t d1, char *fn);
