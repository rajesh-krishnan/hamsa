#ifndef _BUCKET_H_
#define _BUCKET_H_

#include "hdefs.h"

typedef struct {
  int count;
  int arr[BUCKETSIZE];
} Bucket;

void bucket_reset(Bucket *b);
int bucket_add_to(Bucket *b, int id);
int *bucket_get_array(Bucket *b);

#endif /* _BUCKET_H_ */
