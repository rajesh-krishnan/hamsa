#include "dwtahash.h"

DWTAHash *dwtahash_new(int numHashes, int noOfBitsToHash) {
    int *n_array;
    DWTAHash *d = (DWTAHash *) malloc(sizeof(DWTAHash));

    myrnginit();
    d->_numhashes = numHashes;
    d->_rangePow = noOfBitsToHash;
    d->_permute = (int) ceil(numHashes * BINSIZE * 1.0 / noOfBitsToHash);
    d->_lognumhash = (int) ceil(log2(numHashes));
    d->_indices = (int *) malloc(d->_rangePow * d->_permute * sizeof(int));
    d->_pos = (int *) malloc(d->_rangePow * d->_permute * sizeof(int));
    d->_randHash[0] = genrand_int31();
    d->_randHash[1] = genrand_int31();
    if (d->_randHash[0] % 2 == 0) d->_randHash[0]++;
    if (d->_randHash[1] % 2 == 0) d->_randHash[1]++;

    n_array = (int *) malloc(d->_rangePow * sizeof(int));
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

int dwtahash_getRandDoubleHash(DWTAHash *d, int binid, int count) {
    unsigned int tohash = ((binid + 1) << 6) + count;
    return (d->_randHash[0] * tohash << 3) >> (32 - d->_lognumhash);
}

/*
   Expects hashes, values, and hashArray of length d->_numhashes
   Expects xindices of length dataLen if easy is false
   Returns the pointer to hashArray for convenience
*/
static int *gethash(DWTAHash *d, float* data, int dataLen, int *xindices, bool easy) {
    assert(easy == true || xindices != NULL);

    int *hashes = malloc(d->_numhashes * sizeof(int));
    memset(hashes, 0, sizeof(int) * d->_numhashes);
    float *values = malloc(d->_numhashes * sizeof(float));
    memset(values, 0, sizeof(float) * d->_numhashes);
    int *hashArray = malloc(d->_numhashes * sizeof(int));
    memset(hashArray, 0, sizeof(int) * d->_numhashes);

    for (int i = 0; i < d->_numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
    }

    for (int p=0,bin_index=0; p < d->_permute; p++, bin_index+=d->_rangePow) {
        for (int i = 0; i < dataLen; i++) {
            int inner_index = bin_index + (easy ? i : xindices[i]);
            int binid = d->_indices[inner_index];
            float loc_data = data[i];
            if(binid < d->_numhashes && values[binid] < loc_data) {
                values[binid] = loc_data;
                hashes[binid] = d->_pos[inner_index];
            }
        }
    }

    for (int i = 0; i < d->_numhashes; i++)
    {
        int next = hashes[i];
        if (next != INT_MIN)
        {
            hashArray[i] = hashes[i];
            continue;
        }
        int count = 0;
        while (next == INT_MIN)
        {
            count++;
            int r = dwtahash_getRandDoubleHash(d, i, count);
            int index = r < d->_numhashes ? r : d->_numhashes;
            next = hashes[index];
            if (count > 100) break;      /* Densification failure */
        }
        hashArray[i] = next;
    }
    free(hashes);
    free(values);
    return hashArray;
}

int *dwtahash_getHashEasy(DWTAHash *d, float* data, int dataLen) {
    return gethash(d, data, dataLen, NULL, true);
}

int *dwtahash_getHash(DWTAHash *d, int* xindices, float* data, int dataLen) {
    return gethash(d, data, dataLen, xindices, false);
}

