#ifndef _LSH_H_
#define _LSH_H_

#include "bucket.h"
#include "mhelper.h"
#include "hdefs.h"

typedef struct {
    Bucket ** _bucket;
    int _K;
    int _L;
    int _RangePow;
} LSH;

LSH *new_lsh(int K, int L, int RangePow);
void delete_lsh(LSH *l);
void clear_lsh(LSH *l);
void hashesToIndex(LSH *l, int *hashes, int *indices);
void add_lsh(LSH *l, int *indices, int id, int *secondIndices);
void retrieveRaw(LSH *l, int *indices, int **rawResults);

#endif /* _LSH_H_ */
