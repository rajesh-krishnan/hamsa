#include "lsh.h"

LSH *new_lsh(int K, int L, int RangePow) {
    size_t sz  = 1 << RangePow;
    LSH *l     = (LSH *) mymap(sizeof(LSH));
    Bucket *b  = (Bucket *) mymap(L * sz * sizeof(Bucket));
    l->_bucket = (Bucket **) mymap(L * sizeof(Bucket *));
    l->_K = K;
    l->_L = L;
    l->_RangePow = RangePow;
    for (int i = 0; i < L; i++) l->_bucket[i] = &b[i * sz];
    return l;
}

void delete_lsh(LSH *l) {
    size_t sz  = 1 << l->_RangePow;
    Bucket *b  = l->_bucket[0];
    myunmap(b, l->_L * sz * sizeof(Bucket));
    myunmap(l->_bucket, l->_L * sizeof(Bucket *));
    myunmap(l, sizeof(LSH));
}

void clear_lsh(LSH *l) {
    size_t totsz = (1 << l->_RangePow) * l->_L;
    Bucket *b    = l->_bucket[0];
#pragma omp parallel for 
    for (int i = 0; i < totsz; i++) reset(&b[i]);
}


/*
   Expects to be provided indices of length l->_L
*/

void hashesToIndex(LSH *l, int *hashes, int *indices) {
    const int logbinsize = (int)floor(log(BINSIZE));  /* should this be log2? */

    /* int *indices = new int[l->_L]; */
    for (int i = 0; i < l->_L; i++) {
        unsigned int index = 0;
        for (int j = 0; j < l->_K; j++) {
             unsigned int h = hashes[l->_K*i + j];
             index += h<<((l->_K-1-j) * logbinsize);
        }
        indices[i] = index;
    }
}

/*
   Expects to be provided secondIndices of length l->_L
*/
void add_lsh(LSH *l, int *indices, int id, int *secondIndices) {
    /* int * secondIndices = new int[_L]; */
    for (int i = 0; i < l->_L; i++)
    {
        secondIndices[i] = add(&l->_bucket[i][indices[i]], id);
    }
}

/*
  Returns all the buckets
  Expects rawResults of length l->_L
*/

void retrieveRaw(LSH *l, int *indices, int **rawResults) {
    /* int ** rawResults = new int*[_L]; */
    for (int i = 0; i < l->_L; i++)
    {
        rawResults[i] = getAll(&l->_bucket[i][indices[i]]);
    }
}

