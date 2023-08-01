#pragma once
#include "layer.h"

typedef struct _struct_network {
    Layer **_hiddenlayers;
    NodeType *_layersTypes;
    int *_sizesOfLayers;
    float *_Sparsity;
    float _learningRate;
    int _numberOfLayers;
    int _currentBatchSize;
} Network;

/*
Network *network_new(int* sizesOfLayers, NodeType* layersTypes, int noOfLayers, int batchsize, float lr, 
    int inputdim, int* K, int* L, int* RangePow, float* Sparsity, cnpy::npz_t arr);
*/
void network_delete(Network *n);

Layer *getLayer(Network *n, int LayerID);
int predictClass(Network *n, int ** inputIndices, float ** inputValues, int * length, int ** labels, int *labelsize);
int processInput(Network *n, int** inputIndices, float** inputValues, int* lengths, int ** label, int *labelsize, 
    int iter, bool rehash, bool rebuild);

void network_save(Network *n, char *path);
/* XXX: need to be able to load saved network */
