#include "bucket.h"

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

