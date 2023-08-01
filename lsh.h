#ifndef _LSH_H_
#define _LSH_H_
#include "hdefs.h"
#include "mhelper.h"
#include "bucket.h"

typedef struct _struct_lsh {
    Bucket ** _bucket;
    int _K;
    int _L;
    int _RangePow;
} LSH;

LSH *lsh_new(int K, int L, int RangePow);
void lsh_delete(LSH *l);
void lsh_clear(LSH *l);
void lsh_hashes_to_indices(LSH *l, int *hashes, int *indices);
void lsh_add_indices(LSH *l, int *indices, int id, int *secondIndices);
void lsh_retrieve_indices_raw(LSH *l, int *indices, int **rawResults);

#endif /* _LSH_H_ */
