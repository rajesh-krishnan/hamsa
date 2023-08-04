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
#include "klib/khash.h"
#include "sfmt/SFMT.h"

#define MY_PI 3.14159265358979323846 /* pi */

KHASH_MAP_INIT_INT(hist, size_t)

void *mymap(size_t size);
void  myunmap(void *ptr, size_t size);
void  myshuffle(int *array, int n);
float myrand_norm(double mu, double sigma);
int   myrand_unif();
void  mysave_fnpy(float *farr, bool twoD, size_t d0, size_t d1, char *fn);
void  myload_fnpy(float *farr, bool twoD, size_t d0, size_t d1, char *fn);
