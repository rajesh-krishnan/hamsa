#include "dwtahash.h"

static void shuffle(int *array, int n)
{
    if (n <= 1) return;
    for (int i = 0; i < n - 1; i++) {
        int j = i + (genrand_int31() % (n - i + 1));
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

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

#if 0
/* XXX: make these functions not allocate memory */
int *dwtahash_getHashEasy(DWTAHash *d, float* data, int dataLen, int topK) {
    int *hashes = new int[_numhashes];
    float *values = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    for (int i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
    }

    for (int p=0; p< _permute; p++) {
        int bin_index = p * _rangePow;
        for (int i = 0; i < dataLen; i++) {
            int inner_index = bin_index + i;
            int binid = _indices[inner_index];
            float loc_data = data[i];
            if(binid < _numhashes && values[binid] < loc_data) {
                values[binid] = loc_data;
                hashes[binid] = _pos[inner_index];
            }
        }
    }

    for (int i = 0; i < _numhashes; i++)
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
            int index = std::min(
                    getRandDoubleHash(i, count),
                    _numhashes);

            next = hashes[index]; // Kills GPU.
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }
    delete[] hashes;
    delete[] values;
    return hashArray;
}

int *dwtahash_getHash(DWTAHash *d, int* indices, float* data, int dataLen) {
    int *hashes = new int[_numhashes];
    float *values = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    // init hashes and values to INT_MIN to start
    for (int i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
    }

    //
    for (int p = 0; p < _permute; p++) {
        for (int i = 0; i < dataLen; i++) {
            int binid = _indices[p * _rangePow + indices[i]];
            if(binid < _numhashes) {
                if (values[binid] < data[i]) {
                    values[binid] = data[i];
                    hashes[binid] = _pos[p * _rangePow + indices[i]];
                }
            }
        }
    }

    for (int i = 0; i < _numhashes; i++)
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
            int index = std::min(
                    getRandDoubleHash(i, count),
                    _numhashes);

            next = hashes[index]; // Kills GPU.
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }

    delete[] hashes;
    delete[] values;

    return hashArray;
}

#endif
