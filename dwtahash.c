#include "dwtahash.h"

DWTAHash *dwtahash_new(int numHashes, int noOfBitsToHash) {
    int *n_array;
    DWTAHash *d = (DWTAHash *) malloc(sizeof(DWTAHash));
    d->_numhashes = numHashes;
    d->_rangePow = noOfBitsToHash;
    d->_permute = (int) ceil(numHashes * BINSIZE * 1.0 / noOfBitsToHash);
    d->_lognumhash = (int) ceil(log2(numHashes));
    d->_randHash[0] = myrand_unif();
    d->_randHash[1] = myrand_unif();
    if (d->_randHash[0] % 2 == 0) d->_randHash[0]++;
    if (d->_randHash[1] % 2 == 0) d->_randHash[1]++;
    d->_indices = (int *) malloc(d->_rangePow * d->_permute * sizeof(int));
    d->_pos = (int *) malloc(d->_rangePow * d->_permute * sizeof(int));
    n_array = (int *) malloc(d->_rangePow * sizeof(int));
    assert((d->_indices != NULL) && (d->_pos != NULL) && (n_array != NULL));
    for (int i = 0; i < d->_rangePow; i++) n_array[i] = i;
    for (int p = 0; p < d->_permute ;p++) {
        myshuffle(n_array, d->_rangePow);
        int base = d->_rangePow * p;
        for (int j = 0; j < d->_rangePow; j++) {
            d->_indices[base + n_array[j]] = (base + j) / BINSIZE;
            d->_pos[base + n_array[j]] = (base + j) % BINSIZE;
        }
    }
    free(n_array);
    return d;
}

void dwtahash_delete(DWTAHash *d) {
    free(d->_indices);
    free(d->_pos);
    free(d);
}

inline static int __attribute__((always_inline)) dwtahash_getRandDoubleHash(DWTAHash *d, int binid, int count) {
    unsigned int tohash = ((binid + 1) << 6) + count;
    return (d->_randHash[0] * tohash << 3) >> (32 - d->_lognumhash);
}

inline static int __attribute__((always_inline)) *gethash(DWTAHash *d, float* data, int dLen, int *xidx, bool easy) {
    int *hashes = (int *) malloc(d->_numhashes * sizeof(int));
    float *values = (float *) malloc(d->_numhashes * sizeof(float));
    int *hashArray = (int *) malloc(d->_numhashes * sizeof(int));
    for (int i = 0; i < d->_numhashes; i++) hashes[i] = values[i] = INT_MIN;
    memset(hashArray, 0, sizeof(int) * d->_numhashes);

    for (int p=0; p < d->_permute; p++) {
        int bin_index = p * d->_rangePow;
        for (int i = 0; i < dLen; i++) {
            int inner_index = bin_index + (easy ? i : xidx[i]);
            int binid = d->_indices[inner_index];
            float loc_data = data[i];
            if(binid < d->_numhashes && values[binid] < loc_data) {
                values[binid] = loc_data;
                hashes[binid] = d->_pos[inner_index];
            }
        }
    }

    for (int i = 0; i < d->_numhashes; i++) {
        int next = hashes[i];
        if (next != INT_MIN) {
            hashArray[i] = hashes[i];
            continue;
        }
        int count = 0;
        while (next == INT_MIN) {
            count++;
            int r = dwtahash_getRandDoubleHash(d, i, count);
            int index = r < d->_numhashes ? r : d->_numhashes;
            next = hashes[index];
            if (count > 100) break;      /* Densification failure XXX: but why 100? */
        }
        hashArray[i] = next;
    }

    free(hashes);
    free(values);
    return hashArray;
}

int *dwtahash_getHashEasy(DWTAHash *d, float* data, int dLen) { return gethash(d, data, dLen, NULL, true); }
int *dwtahash_getHash(DWTAHash *d, int *xndx, float* data, int dLen) { return gethash(d, data, dLen, xndx, false); }

