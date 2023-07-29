#ifndef _BUCKET_H_
#define _BUCKET_H_

#include "hdefs.h"

typedef struct {
  int inited;
  int count;
  int arr[BUCKETSIZE];
} Bucket;

void reset(Bucket *b);
int add(Bucket *b, int id);
int retrieve(Bucket *b, int index);
int *getAll(Bucket *b);
void test_bucket();

#endif /* _BUCKET_H_ */
