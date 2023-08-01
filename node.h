#ifndef _NODE_H_
#define _NODE_H_
#include "mhelper.h"

typedef enum { ReLU, Softmax} NodeType;

typedef struct _struct_train {
    float _lastDeltaforBPs;
    float _lastActivations;
    float _lastGradients;
    int _ActiveinputIds;
} __attribute__ ((aligned (64))) Train;

typedef struct _struct_node {
    size_t _dim;
    size_t _layerNum;
    size_t _IDinLayer;
    NodeType _type;
    int _currentBatchsize;
    float *_weights;
    float _bias;
    float *_adamAvgMom;
    float *_adamAvgVel;
    float* _t;
    Train *_train;
    int _activeInputs;
    float _tbias;
    float _adamAvgMombias;
    float _adamAvgVelbias;
    int *_indicesInTables;
    int *_indicesInBuckets;
    int *_update;
} Node;

void node_update(Node *n, int dim, int nodeID, int layerID, NodeType type, int batchsize, 
    float *weights, float bias, float *adamAvgMom, float *adamAvgVel, float *adam_t, 
    Train* train_blob);
float node_get_last_activation(Node *n, int inputID);
void node_set_last_activation(Node *n, int inputID, float realActivation);
void node_increment_delta(Node *n, int inputID, float incrementValue);
float node_get_activation(Node *n, int* indices, float* values, int length, int inputID);
bool node_get_input_active(Node *n, int inputID);
bool node_get_active_inputs(Node *n);
float node_perturb_weight(Node *n, int weightid, float delta);
float node_get_gradient(Node *n, int weightid, int inputID, float InputVal);
void node_compute_softmax_stats(Node *n, float normalizationConstant, int inputID, int* label, int labelsize);
void node_backprop(Node *n, Node* previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, 
    float learningRate, int inputID);
void node_backprop_firstlayer(Node *n, int* nnzindices, float* nnzvalues, int nnzSize, 
    float learningRate, int inputID);

#endif /* _NODE_H_ */
