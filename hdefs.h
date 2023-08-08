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

#include "uthash/uthash.h"
#include "hamsa.h"

typedef struct _struct_histo {
    int key;
    size_t value;
    UT_hash_handle hh;         /* makes this structure hashable */
} Histo;

#define MY_PI 3.14159265358979323846 /* pi */
#define BETA1 0.9
#define BETA2 0.999
#define EPS 0.00000001
#define MINACTIVE 1000
#define THRESH 0

/* Internal functions */

void *mymap(size_t size);
void myunmap(void *ptr, size_t size);
void myshuffle(int *array, int n);
float myrand_norm(double mu, double sigma);
int myrand_unif();
void mysave_fnpy(float *farr, size_t d0, size_t d1, char *fn);
void myload_fnpy(float *farr, size_t d0, size_t d1, char *fn);
void ht_put(Histo **counts, int key, size_t value);
void ht_incr(Histo **counts, int key);
void ht_delkey(Histo **counts, int key);
void ht_del(Histo **counts, Histo **cur);
void ht_destroy(Histo **counts);
unsigned int ht_size(Histo **counts);

DWTAHash *dwtahash_new(int numHashes, int noOfBitsToHash);
void dwtahash_delete(DWTAHash *d);
int *dwtahash_getHashEasy(DWTAHash *d, float* data, int dLen);
int *dwtahash_getHash(DWTAHash *d, int *xndx, float* data, int dLen);

LSHT *lsht_new(int K, int L, int RangePow);
void lsht_delete(LSHT *l);
void lsht_clear(LSHT *l);
void lsht_add(LSHT *l, int *hashes, int id);
void lsht_retrieve_histogram(LSHT *l, int *hashes, Histo **counts);

void node_update(Node *n, int nodeID, NodeType type, int batchsize, 
    float *weights, float *bias, float *adamAvgMom, float *adamAvgVel, float *adam_t, Train* train_blob);
float node_get_last_activation(Node *n, int inputID);
void node_set_last_activation(Node *n, int inputID, float realActivation);
void node_increment_delta(Node *n, int inputID, float incrementValue);
float node_get_activation(Node *n, int* indices, float* values, int length, int inputID);
bool node_get_input_active(Node *n, int inputID);
bool node_get_active_inputs(Node *n);
void node_compute_softmax_stats(Node *n, float normalizationConstant, int inputID, int* label, int labelsize);
void node_backprop(Node *n, Node* prevLayerNodes, int* prevLayerActiveNodeIds, int prevLayerActiveNodeSize, 
    float learningRate, int inputID);
void node_backprop_firstlayer(Node *n, int* nnzindices, float* nnzvalues, int nnzSize, 
    float learningRate, int inputID);
void node_adam(Node *n, int dim, float tmplr, int ratio);

