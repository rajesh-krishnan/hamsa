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

/*
int* add(LSH *l, int *indices, int id);
int add(LSH *l, int indices, int tableId, int id);
int * hashesToIndex(LSH *l, int * hashes);
void hashesToIndexAddOpt(LSH *l, int * hashes, int id);
int** retrieveRaw(LSH *l, int *indices);
int retrieve(LSH *l, int table, int indices, int bucket);
void count(LSH *l);
*/

#endif /* _LSH_H_ */
