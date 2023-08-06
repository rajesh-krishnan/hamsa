#include "hdefs.h"

static int logbinsize = (int)floor(log2(BINSIZE));  /* original used natural log, check */

inline static void __attribute__((always_inline)) bucket_reset(Bucket *b) { b->count = 0; }

inline static int __attribute__((always_inline)) bucket_add_to(Bucket *b, int id) {
    int index = b->count & (BUCKETSIZE - 1);        /* place in [0, BUCKETSIZE), cheaper than modulo */
    b->arr[index] = id;
    b->count++;
    assert ((id > 0) && (b->count > 0));            /* only positive values, and no integer overflow */
    return index;
}

inline static int __attribute__((always_inline)) *bucket_get_array(Bucket *b) {
    if (b->count<BUCKETSIZE) b->arr[b->count]=-1;  /* set first unused entry in bucket to -1 */
    return b->arr;
}

LSHT *lsht_new(int K, int L, int RangePow) {
    assert (K * logbinsize == RangePow);
    Bucket *b;
    size_t sz  = 1 << RangePow;
    LSHT *l     = (LSHT *) malloc(sizeof(LSHT));
    l->_bucket = (Bucket **) malloc(L * sizeof(Bucket *));
    assert((l != NULL) && (l->_bucket !=NULL));
    b = (Bucket *) mymap(L * sz * sizeof(Bucket));
    for (int i = 0; i < L; i++) l->_bucket[i] = &b[i * sz];
    l->_K = K;
    l->_L = L;
    l->_RangePow = RangePow;
    lsht_clear(l);
    return l;
}

void lsht_delete(LSHT *l) {
    size_t sz  = 1 << l->_RangePow;
    Bucket *b  = l->_bucket[0];
    myunmap(b, l->_L * sz * sizeof(Bucket));
    free(l->_bucket);
    free(l);
}

void lsht_clear(LSHT *l) { memset(l->_bucket[0], 0, (1 << l->_RangePow) * l->_L * sizeof(Bucket)); }

inline static unsigned int __attribute__((always_inline)) ith_index(LSHT *l, int *hashes, int i) {
    unsigned int index = 0;
    for (int j = 0; j < l->_K; j++) {
        unsigned int h = hashes[l->_K*i + j];
        index += h<<((l->_K-1-j) * logbinsize);
    }
    return index;
}

void lsht_add(LSHT *l, int *hashes, int id) {
    for (int i = 0; i < l->_L; i++) {
        unsigned int index = ith_index(l, hashes, i);
	bucket_add_to(&l->_bucket[i][index], id + 1);           /* incr 1, 0 is bad for densification */
    }                                                           /* reverse upon retrieval */
}

/* Collect items and their counts across all retrieves buckets, put in hashtable */
void lsht_retrieve_histogram(LSHT *l, int *hashes, khash_t(hist) *h) {
    int isnew, *arr;
    khiter_t k;
    for (int i = 0; i < l->_L; i++) {
        unsigned int index = ith_index(l, hashes, i);
        arr = bucket_get_array(&l->_bucket[i][index]);
        for (int j = 0; j < BUCKETSIZE; j++) {
           if (arr[j] < 0) break;                               /* bucket array terminated by -1 */
           k = kh_put(hist, h, arr[j] - 1, &isnew);             /* add to h, isnew is 1 if new else 0 */
                                                                /* -1 to get index, reverse +1 from lsht_add */
           kh_value(h, k) = (isnew) ? 1 : (kh_value(h, k) + 1); /* count set to 1 if new, else incremented by 1 */
        }
    }
}

