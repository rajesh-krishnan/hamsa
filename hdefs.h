#pragma once
#include <omp.h>
#include <math.h>
#include <time.h>
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

#include "npy_array/npy_array.h"
#include "uthash/uthash.h"
#include "sfmt/SFMT.h"
#include "hamsa.h"

#define MY_PI 3.14159265358979323846 /* pi */
#define BETA1 0.9
#define BETA2 0.999
#define EPS 0.00000001
#define MINACTIVE 1000
#define THRESH 0

typedef struct _struct_histo {
    int key;
    size_t value;
    UT_hash_handle hh;         /* makes this structure hashable */
} Histo;

/* Internal functions and signatures */

inline static void __attribute__((always_inline)) myunmap(void *ptr, size_t size) { 
    munmap(ptr, size); 
}

inline static void __attribute__((always_inline)) *mymap(size_t size) {
    void *ptr;
    ptr = mmap(NULL, size, PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (ptr != MAP_FAILED) return ptr;
    fprintf(stderr, "hugetlb allocation failed %ld\n", size);
    ptr = mmap(NULL, size, PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr != MAP_FAILED) return ptr;
    fprintf(stderr, "mmap failed \n");
    exit(0);
}

inline static void __attribute__((always_inline)) mysave_fnpy(float *farr, size_t d0, size_t d1, char *fn) {
    npy_array_save( fn, NPY_ARRAY_BUILDER( farr, SHAPE( d0, d1 ), NPY_DTYPE_FLOAT32 ) );
}

inline static void __attribute__((always_inline)) myload_fnpy(float *farr, size_t d0, size_t d1, char *fn) {
    npy_array_t *a = npy_array_load(fn);
    assert(a != NULL);
    assert((a->typechar == 'f') && (a->elem_size == sizeof(float)) && (a->ndim == 2));
    assert(a->shape[0] == d0);
    assert(a->shape[1] == d1);
    memcpy(farr, a->data, d0 * d1 * sizeof(float));
    npy_array_free(a);
}

inline static void __attribute__((always_inline)) ht_put(Histo **counts, int key, size_t value) {
    Histo *s;
    HASH_FIND_INT(*counts, &key, s);  
    if (s == NULL) {
        s = (Histo *)malloc(sizeof *s);
        s->key = key;
        HASH_ADD_INT(*counts, key, s);
    } 
    s->value = value;
}

inline static void __attribute__((always_inline)) ht_incr(Histo **counts, int key) {
    Histo *s;
    HASH_FIND_INT(*counts, &key, s);  
    if (s == NULL) {
        s = (Histo *)malloc(sizeof *s);
        s->key = key;
        HASH_ADD_INT(*counts, key, s);
        s->value = 1;
    } else {
        s->value += 1;
    }
}

inline static void __attribute__((always_inline)) ht_delkey(Histo **counts, int key) {
    Histo *s;
    HASH_FIND_INT(*counts, &key, s);  
    if (s != NULL) {
        HASH_DEL(*counts, s);
        free(s);
    }
}

inline static void __attribute__((always_inline)) ht_del(Histo **counts, Histo **cur) {
    HASH_DEL(*counts, *cur); 
    free(*cur);
}

inline static void __attribute__((always_inline)) ht_destroy(Histo **counts) {
    Histo *cur, *tmp;
    HASH_ITER(hh, *counts, cur, tmp) {
        HASH_DEL(*counts, cur);
        free(cur);
    }
}

inline static unsigned int __attribute__((always_inline)) ht_size(Histo **counts) { 
    return HASH_COUNT(*counts); 
}

int myrand_unif();
float myrand_norm(double mu, double sigma);
void myrand_shuffle(int *array, int n);

DWTAHash *dwtahash_new(int numHashes, int noOfBitsToHash);
void dwtahash_delete(DWTAHash *d);
int *dwtahash_getHashEasy(DWTAHash *d, float *data, int dLen);
int *dwtahash_getHash(DWTAHash *d, int *xndx, float *data, int dLen);

LSHT *lsht_new(int K, int L, int RangePow);
void lsht_delete(LSHT *l);
void lsht_clear(LSHT *l);
void lsht_add(LSHT *l, int *hashes, int id);
void lsht_retrieve_histogram(LSHT *l, int *hashes, Histo **counts);

void node_update(Node *n, int nodeID, NodeType type, int batchsize, Train *train_blob,
    float *weights, float *adamAvgMom, float *adamAvgVel, float *adam_t,
    float *bias, float *adamAvgMombias, float *adamAvgVelbias);
float node_get_last_activation(Node *n, int inputID);
void node_set_last_activation(Node *n, int inputID, float realActivation);
void node_increment_delta(Node *n, int inputID, float incrementValue);
float node_get_activation(Node *n, int *indices, float *values, int length, int inputID);
bool node_get_input_active(Node *n, int inputID);
void node_compute_softmax_stats(Node *n, float normalizationConstant, int inputID, int batchsize, int *label, int labelsize);
void node_backprop(Node *n, Node *prevLayerNodeArray, int *prevLayerActiveNodeIds, int prevLayerActiveNodeSize, 
    float learningRate, int inputID);
void node_backprop_firstlayer(Node *n, int *nnzindices, float *nnzvalues, int nnzSize, 
    float learningRate, int inputID);
void node_adam(Node *n, int dim, int batchsize, float tmplr, int ratio);

