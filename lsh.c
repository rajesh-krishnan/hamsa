#include "lsh.h"

static int logbinsize = (int)floor(log2(BINSIZE));  /* XXX: original code used log, not log2 */

void bucket_reset(Bucket *b) { b->count = 0; }

int bucket_add_to(Bucket *b, int id) {
    int index = b->count & (BUCKETSIZE - 1);  /* place index in [0, BUCKETSIZE), cheaper than modulo */
    b->arr[index] = id;
    b->count++;
    assert (b->count > 0);                    /* check for overflow */
    return index;
}

int *bucket_get_array(Bucket *b) {
    if (b->count<BUCKETSIZE) b->arr[b->count]=-1;  /* set first unused entry in bucket to -1 */
    return b->arr;
}

LSH *lsh_new(int K, int L, int RangePow) {
    assert (K * logbinsize == RangePow);
    Bucket *b;
    size_t sz  = 1 << RangePow;
    LSH *l     = (LSH *) mymap(sizeof(LSH));
    l->_bucket = (Bucket **) mymap(L * sizeof(Bucket *));
    b = (Bucket *) mymap(L * sz * sizeof(Bucket));            /* this is really large */
    for (int i = 0; i < L; i++) l->_bucket[i] = &b[i * sz];
    l->_K = K;
    l->_L = L;
    l->_RangePow = RangePow;
    return l;
}

void lsh_delete(LSH *l) {
    size_t sz  = 1 << l->_RangePow;
    Bucket *b  = l->_bucket[0];
    myunmap(b, l->_L * sz * sizeof(Bucket));
    myunmap(l->_bucket, l->_L * sizeof(Bucket *));
    myunmap(l, sizeof(LSH));
}

void lsh_clear(LSH *l) {
    size_t totsz = (1 << l->_RangePow) * l->_L;
    Bucket *b    = l->_bucket[0];
#pragma omp parallel for 
    for (int i = 0; i < totsz; i++) bucket_reset(&b[i]);
}

/*
   Expects hashes of length l->_L * l->_K
*/
void lsh_hashes_to_indices_add(LSH *l, int *hashes, int id) {
    for (int i = 0; i < l->_L; i++) {
        unsigned int index = 0;
        for (int j = 0; j < l->_K; j++) {
             unsigned int h = hashes[l->_K*i + j];
             index += h<<((l->_K-1-j) * logbinsize);
        }
	bucket_add_to(&l->_bucket[i][index], id);
    }
}

/*
  Expects indices and rawResults of length l->_L
  Returns L arrays of buckets at i,indices[i] locations in rawResults[i]
*/
void lsh_retrieve_indices_raw(LSH *l, int *indices, int **rawResults) {
    for (int i = 0; i < l->_L; i++) 
        rawResults[i] = bucket_get_array(&l->_bucket[i][indices[i]]);
}

