#ifndef _NODE_H_
#define _NODE_H_

#include "hdefs.h"

typedef enum { ReLU, Softmax} NodeType;

typedef struct {
    float _lastDeltaforBPs;
    float _lastActivations;
    float _lastGradients;
    int _ActiveinputIds;
} __attribute__ ((aligned (64))) Train;

typedef struct {
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
    float _mirrorbias;
    float _tbias;
    float _adamAvgMombias;
    float _adamAvgVelbias;
    float *_mirrorWeights;

    int *_indicesInTables;
    int *_indicesInBuckets;
    int *_update;
} Node;

void Update(Node *n, int dim, int nodeID, int layerID, NodeType type, int batchsize, 
            float *weights, float bias, float *adamAvgMom, float *adamAvgVel, float *adam_t, 
            Train* train_blob);
/*
float getLastActivation(Node *n, int inputID);
void incrementDelta(Node *n, int inputID, float incrementValue);
float getActivation(Node *n, int* indices, float* values, int length, int inputID);
bool getInputActive(Node *n, int inputID);
bool getActiveInputs(Node *n, void);
void SetlastActivation(Node *n, int inputID, float realActivation);
void ComputeExtaStatsForSoftMax(Node *n, float normalizationConstant, int inputID, int* label, int labelsize);
void backPropagate(Node *n, Node* previousNodes,int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID);
void backPropagateFirstLayer(Node *n, int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID);
float perturbWeight(Node *n, int weightid, float delta);
float getGradient(Node *n, int weightid, int inputID, float InputVal);
*/

#endif /* _NODE_H_ */
