#pragma once
#include "myhelper.h"

typedef struct _struct_bucket {
  int count;
  int arr[BUCKETSIZE];
} Bucket;

typedef struct _struct_lsh {
    Bucket ** _bucket;
    int _K;
    int _L;
    int _RangePow;
} LSH;

LSH *lsh_new(int K, int L, int RangePow);
void lsh_delete(LSH *l);
void lsh_clear(LSH *l);
void lsh_add(LSH *l, int *hashes, int id);
void lsh_retrieve_raw(LSH *l, int *hashes, int **rawResults);
void lsh_retrieve_histogram(LSH *l, int *hashes, khash_t(hist) *h);
