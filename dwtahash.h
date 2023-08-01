#ifndef _DWTAHASH_H_
#define _DWTAHASH_H_
#include "hdefs.h"
#include "mhelper.h"

typedef struct _struct_dwtahash {
    int _numhashes;
    int _rangePow;
    int _lognumhash;
    int _permute;
    int _randHash[2];
    int *_indices;
    int *_pos;
} DWTAHash;

DWTAHash *dwtahash_new(int numHashes, int noOfBitsToHash);
void dwtahash_delete(DWTAHash *d);
int dwtahash_getRandDoubleHash(DWTAHash *d, int binid, int count);
int *dwtahash_getHashEasy(DWTAHash *d, float* data, int dataLen, int topK,
    int *hashes, float *values, int *hashArray);
int *dwtahash_getHash(DWTAHash *d, int* xindices, float* data, int dataLen,
    int *hashes, float *values, int *hashArray);

#endif /* _DWTAHASH_H_ */
