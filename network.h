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
void network_save(Network *n, char *path);
void network_load(Network *n, char *path);
Layer *network_getLayer(Network *n, int LayerID);
int network_infer(Network *n, int **inputIndices, float **inputValues, int *length, int **labels, int *labelsize);
void network_train(Network *n, int **inputIndices, float **inputValues, int *lengths, int **label, int *labelsize, 
    int iter, bool rehash, bool rebuild);

