#include "dwtahash.h"

DWTAHash *dwtahash_new(int numHashes, int noOfBitsToHash) {
    int *n_array;
    unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
    DWTAHash *d = (DWTAHash *) mymap(sizeof(DWTAHash));

    init_by_array(init, length);
    d->_numhashes = numHashes;
    d->_rangePow = noOfBitsToHash;
    d->_permute = (int) ceil(numHashes * BINSIZE * 1.0 / noOfBitsToHash);
    d->_lognumhash = (int) ceil(log2(numHashes));
    d->_indices = (int *) mymap (d->_rangePow * d->_permute * sizeof(int));
    d->_pos = (int *) mymap (d->_rangePow * d->_permute * sizeof(int));
    d->_randHash[0] = genrand_int31();
    d->_randHash[1] = genrand_int31();
    if (d->_randHash[0] % 2 == 0) d->_randHash[0]++;
    if (d->_randHash[1] % 2 == 0) d->_randHash[1]++;

    n_array = (int *) mymap (d->_rangePow * sizeof(int));
    for (int i = 0; i < d->_rangePow; i++) n_array[i] = i;

    for (int p = 0; p < d->_permute ;p++) {
        shuffle(n_array, d->_rangePow);
        int base = d->_rangePow * p;
        for (int j = 0; j < d->_rangePow; j++) {
            d->_indices[base + n_array[j]] = (base + j) / BINSIZE;
            d->_pos[base + n_array[j]] = (base + j) % BINSIZE;
        }
    }

    myunmap (n_array, d->_rangePow * sizeof(int));
    return d;
}

void dwtahash_delete(DWTAHash *d) {
    myunmap(d->_indices, d->_rangePow * d->_permute * sizeof(int));
    myunmap(d->_pos, d->_rangePow * d->_permute * sizeof(int));
    myunmap(d, sizeof(DWTAHash));
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
static int *gethash(DWTAHash *d, float* data, int dataLen, int *hashes, float *values, 
    int *xindices, bool easy, int *hashArray) {
    assert(easy == true || xindices != NULL);

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
    return hashArray;
}

int *dwtahash_getHashEasy(DWTAHash *d, float* data, int dataLen, int topK,
    int *hashes, float *values, int *hashArray) {
    return gethash(d, data, dataLen, hashes, values, NULL, false, hashArray);
}

int *dwtahash_getHash(DWTAHash *d, int* xindices, float* data, int dataLen,
    int *hashes, float *values, int *hashArray) {
    return gethash(d, data, dataLen, hashes, values, xindices, true, hashArray);
}

