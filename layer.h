#pragma once
#include "lsh.h"
#include "node.h"
#include "dwtahash.h"

typedef struct _struct_layer {
    size_t _noOfNodes;
    int _previousLayerNumOfNodes;
    int _layerID;
    NodeType _type;
    int _batchsize;
    int _K;
    int _L;
    int _RangePow;
    Node *_Nodes;
    LSH *_hashTables;
    DWTAHash *_dwtaHasher;
    Train *_train_array;
    int *_randNode;
    float *_weights;
    float *_bias;
    float *_adamAvgMom;
    float *_adamAvgVel;
    float *_adamT;
    float *_normalizationConstants;
} Layer;

Layer *layer_new(size_t noOfNodes, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, 
    int K, int L, int RangePow, bool load, char *path);
void layer_delete(Layer *l);
void layer_randinit(Layer *l);
void layer_load(Layer *l, char *path);
void layer_save(Layer *l, char *path);

Node *layer_getNodebyID(Layer *l, size_t nodeID);
Node *layer_getAllNodes(Layer *l);
int layer_getNodeCount(Layer *l);
float layer_getNormalizationConstant(Layer *l, int inputID);
void layer_updateTable(Layer *l);
void layer_updateRandomNodes(Layer *l);
void layer_addToHashTable(Layer *l, float* weights, int length, int id);
int layer_queryActiveNodeandComputeActivations(Layer *l, int **activenodesperlayer, float **activeValuesperlayer, 
    int *length, int layerIndex, int inputID,  int *label, int labelsize, float Sparsity, int iter);
