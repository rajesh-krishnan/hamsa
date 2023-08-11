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

inline static size_t __attribute__((always_inline)) lsht_size(int K, int L, int RangePow) {
    size_t sz  = (1 << RangePow);
    size_t hugepg_size = (1L << 21);  /* 2MB Hugepage */
    size_t buffer_size = sizeof(LSHT) + L * sizeof (Bucket *) + L * sz * sizeof(Bucket);
    return (size_t) ceil(hugepg_size * buffer_size * 1.0 / hugepg_size);
}

LSHT *lsht_new(int K, int L, int RangePow) {
    assert (K * logbinsize == RangePow);
    size_t sz  = 1 << RangePow;
    size_t buffer_size  = lsht_size(K, L, RangePow);
    void *buf = mymap(buffer_size);
    fprintf(stderr, "Allocated %ld bytes for LSHT at %p\n", buffer_size, buf);

    LSHT *l    = (LSHT *)    buf; buf += sizeof(LSHT);
    l->_bucket = (Bucket **) buf; buf += L * sizeof(Bucket *);
    Bucket *b  = (Bucket *)  buf; buf += L * sz * sizeof(Bucket);

    for (int i = 0; i < L; i++) l->_bucket[i] = &b[i * sz];
    l->_K = K;
    l->_L = L;
    l->_RangePow = RangePow;
    lsht_clear(l);
    return l;
}

void lsht_delete(LSHT *l) {
    Bucket *b  = l->_bucket[0];
    size_t buffer_size = lsht_size(l->_K, l->_L, l->_RangePow);
    fprintf(stderr, "Freeing %ld bytes for LSHT  at %p\n", buffer_size, l);
    myunmap(l, buffer_size);
}

void lsht_clear(LSHT *l) { memset(l->_bucket[0], 0, (1 << l->_RangePow) * l->_L * sizeof(Bucket)); }

inline static unsigned int __attribute__((always_inline)) ith_index(LSHT *l, int *hashes, int i) {
    unsigned int index = 0;
    for (int j = 0; j < l->_K; j++) {
        unsigned int h = hashes[l->_K*i + j];
        if(h == INT_MIN) continue;                     /* dwtahash returns INT_MIN on densification failure, yikes! */
        assert((h >= 0) && (h < (1 << l->_RangePow)));
        index += h<<((l->_K-1-j) * logbinsize);
    }
    assert(index < (1 << l->_RangePow));
    return index;
}

void lsht_add(LSHT *l, int *hashes, int id) {
    assert ((id >= 0) && (id < 0x7fffffff));
    for (int i = 0; i < l->_L; i++) {
        unsigned int index = ith_index(l, hashes, i);
	bucket_add_to(&l->_bucket[i][index], id + 1);           /* incr 1, 0 bad for hashing? */
    }                                                           /* reverse upon retrieval */
}

/* Collect items and their counts across all retrieves buckets, put in hashtable */
void lsht_retrieve_histogram(LSHT *l, int *hashes, Histo **counts) {
    int *arr;
    for (int i = 0; i < l->_L; i++) {
        unsigned int index = ith_index(l, hashes, i);
        arr = bucket_get_array(&l->_bucket[i][index]);
        for (int j = 0; j < BUCKETSIZE; j++) {
           if (arr[j] < 0) break;                               /* bucket array terminated by -1 */
           ht_incr(counts, arr[j] - 1);                         /* incr count of item, initialize if needed */
                                                                /* arr[j] -1 reverses +1 in lsht_add */
        }
    }
}

