#pragma once
#include "myhelper.h"

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
int *dwtahash_getHashEasy(DWTAHash *d, float* data, int dLen);
int *dwtahash_getHash(DWTAHash *d, int *xndx, float* data, int dLen);
