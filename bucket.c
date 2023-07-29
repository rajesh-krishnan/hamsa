#include "bucket.h"

void reset(Bucket *b) {
    b->inited = -1;
    b->count = 0;
}

int add(Bucket *b, int id) {
    b->inited += 1;
    int index = b->count & (BUCKETSIZE - 1);
    b->arr[index] = id;
    b->count++;
    return index;
}

int retrieve(Bucket *b, int index) { return (index >= BUCKETSIZE) ? -1 : b->arr[index]; }

int getSize(Bucket *b) { return b->count; }

int *getAll(Bucket *b) {
    if (b->inited == -1) return NULL;
    if (b->count<BUCKETSIZE) b->arr[b->count]=-1; 
    return b->arr;
}

void test_bucket() {
    Bucket * b = malloc(sizeof(Bucket));
    reset(b);
    add(b,3);
    printf("Size: %d\n", getSize(b));
    printf("Retrieved: %d\n", retrieve(b,0));
    reset(b);
    printf("Size: %d\n", getSize(b));
}

