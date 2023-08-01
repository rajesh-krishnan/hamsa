#pragma once
#include "lsh.h"
#include "node.h"
#include "dwtahash.h"

typedef struct _struct_layer {
    NodeType _type;
    Node *_Nodes;
    int *_randNode;
    float *_normalizationConstants;
    int _K;
    int _L;
    int _RangeRow;
    int _previousLayerNumOfNodes;
    int _batchsize;
    Train *_train_array;

    int _layerID;
    int _noOfActive;
    size_t _noOfNodes;
    float* _weights;
    float* _adamAvgMom;
    float* _adamAvgVel;
    float* _bias;
    LSH *_hashTables;
    DWTAHash *_dwtaHasher;
    int * _binids;
} Layer;

Layer *layer_new(size_t _numNodex, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, 
    int K, int L, int RangePow, float Sparsity, float* weights, float* bias, float *adamAvgMom, float *adamAvgVel);
void layer_delete(Layer *l);
void layer_save(Layer *l, char *path);

Node *getNodebyID(Layer *l, size_t nodeID);
Node *getAllNodes(Layer *l);
int getNodeCount(Layer *l);
void addtoHashTable(Layer *l, float* weights, int length, float bias, int id);
float getNormalizationConstant(Layer *l, int inputID);
void updateTable(Layer *l);
void updateRandomNodes(Layer *l);
int queryActiveNodeandComputeActivations(Layer *l, int** activenodesperlayer, float** activeValuesperlayer, 
    int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
