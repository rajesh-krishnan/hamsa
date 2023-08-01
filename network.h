#pragma once
#include "layer.h"

typedef struct _struct_network {
    Layer **_hiddenlayers;
    int _numberOfLayers;
    float _learningRate;
    int _currentBatchSize;
} Network;

Network *network_new(int *sizesOfLayers, NodeType *layersTypes, int noOfLayers, int batchSize, float lr, 
    int inputdim, int *K, int *L, int *RangePow, float *Sparsity, bool load, char *path);
void network_delete(Network *n);

Layer *getLayer(Network *n, int LayerID);
int predictClass(Network *n, int ** inputIndices, float ** inputValues, int * length, int ** labels, int *labelsize);
int processInput(Network *n, int** inputIndices, float** inputValues, int* lengths, int ** label, int *labelsize, 
    int iter, bool rehash, bool rebuild);

void network_save(Network *n, char *path);
/* XXX: need to be able to load saved network */
