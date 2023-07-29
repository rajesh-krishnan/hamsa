#include "bucket.h"

void reset(Bucket *b) {
    b->inited = -1;
    b->count = 0;
}

int add(Bucket *b, int id) {
    int index = b->count & (BUCKETSIZE - 1);
    b->arr[index] = id;
    b->inited = 1;
    b->count++;
    return index;
}

int retrieve(Bucket *b, int index) { return (index >= BUCKETSIZE) ? -1 : b->arr[index]; }

int *getAll(Bucket *b) {
    if (b->inited == -1) return NULL;
    if (b->count<BUCKETSIZE) b->arr[b->count]=-1; 
    return b->arr;
}

