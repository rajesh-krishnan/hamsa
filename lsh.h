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

void bucket_reset(Bucket *b);
int bucket_add_to(Bucket *b, int id);
int *bucket_get_array(Bucket *b);

LSH *lsh_new(int K, int L, int RangePow);
void lsh_delete(LSH *l);
void lsh_clear(LSH *l);
void lsh_hashes_to_indices_add(LSH *l, int *hashes, int id);
void lsh_retrieve_indices_raw(LSH *l, int *indices, int **rawResults);
