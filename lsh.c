#include "lsh.h"

LSH *new_lsh(int K, int L, int RangePow) {
    size_t sz  = 1 << RangePow;
    LSH *l     = (LSH *) mymap(sizeof(LSH));
    Bucket *b  = (Bucket *) mymap(L * sz * sizeof(Bucket));
    l->_bucket = (Bucket **) mymap(L * sizeof(Bucket *));
    l->_K = K;
    l->_L = L;
    l->_RangePow = RangePow;
    for (int i = 0; i < L; i++) l->_bucket[i] = &b[i * sz];
    return l;
}

void delete_lsh(LSH *l) {
    size_t sz  = 1 << l->_RangePow;
    Bucket *b  = l->_bucket[0];
    myunmap(b, l->_L * sz * sizeof(Bucket));
    myunmap(l->_bucket, l->_L * sizeof(Bucket *));
    myunmap(l, sizeof(LSH));
}

void clear_lsh(LSH *l) {
    size_t totsz = (1 << l->_RangePow) * l->_L;
    Bucket *b    = l->_bucket[0];
#pragma omp parallel for 
    for (int i = 0; i < totsz; i++) reset(&b[i]);
}

/*
int *add(LSH *l, int *indices, int id);
int addOne(LSH *l, int indices, int tableId, int id);
int *hashesToIndex(LSH *l, int * hashes);
void hashesToIndexAddOpt(LSH *l, int * hashes, int id);
int **retrieveRaw(LSH *l, int *indices);
int retrieve(LSH *l, int table, int indices, int bucket);
void count(LSH *l);
*/


/*

void LSH::count()
{
    for (int j=0; j<_L;j++) {
        int total = 0;
        for (int i = 0; i < 1 << _RangePow; i++) {
            if (_bucket[j][i].getSize()!=0) {
                cout <<_bucket[j][i].getSize() << " ";
            }
            total += _bucket[j][i].getSize();
        }
        cout << endl;
        cout <<"TABLE "<< j << "Total "<< total << endl;
    }
}


int* LSH::hashesToIndex(int * hashes)
{
  const int logbinsize = (int)floor(log(binsize));

    int * indices = new int[_L];
    for (int i = 0; i < _L; i++)
    {
        unsigned int index = 0;

        for (int j = 0; j < _K; j++)
        {

            if (HashFunction==4){
                unsigned int h = hashes[_K*i + j];
                index += h<<(_K-1-j);
            }else if (HashFunction==1 | HashFunction==2){
                unsigned int h = hashes[_K*i + j];
                index += h<<((_K-1-j) * logbinsize);

            }else {
                unsigned int h = rand1[_K*i + j];
                h *= rand1[_K * i + j];
                h ^= h >> 13;
                h ^= rand1[_K * i + j];
                index += h * hashes[_K * i + j];
            }
        }
        if (HashFunction==3) {
            index = index&((1<<_RangePow)-1);
        }
        indices[i] = index;
    }

    return indices;
}


int* LSH::add(int *indices, int id)
{
    int * secondIndices = new int[_L];
    for (int i = 0; i < _L; i++)
    {
        secondIndices[i] = _bucket[i][indices[i]].add(id);
    }

    return secondIndices;
}

void LSH::hashesToIndexAddOpt(int * hashes, int id) {
  const int logbinsize = (int)floor(log(binsize));
  for (int i = 0; i < _L; i++) {
    unsigned int index = 0;

    for (int j = 0; j < _K; j++) {
      if (HashFunction==4){
        unsigned int h = hashes[_K*i + j];
        index += h<<(_K-1-j);
      } else if (HashFunction==1 | HashFunction==2){
        unsigned int h = hashes[_K*i + j];
        index += h<<((_K-1-j)*logbinsize);
      } else {
        unsigned int h = rand1[_K*i + j];
        h *= rand1[_K * i + j];
        h ^= h >> 13;
        h ^= rand1[_K * i + j];
        index += h * hashes[_K * i + j];
      }
    }
    if (HashFunction==3) {
      index = index&((1<<_RangePow)-1);
    }
    _bucket[i][index].add(id);
  }
}

int LSH::add(int tableId, int indices, int id)
{
    int secondIndices = _bucket[tableId][indices].add(id);
    return secondIndices;
}
*/


/*
* Returns all the buckets
*/
/*
int** LSH::retrieveRaw(int *indices)
{
    int ** rawResults = new int*[_L];

    for (int i = 0; i < _L; i++)
    {
        rawResults[i] = _bucket[i][indices[i]].getAll();
    }
    return rawResults;
}


int LSH::retrieve(int table, int indices, int bucket)
{
    return _bucket[table][indices].retrieve(bucket);
}

*/
