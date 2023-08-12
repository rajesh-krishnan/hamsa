#include "hdefs.h"

void node_update(Node *n, int nodeID, NodeType type, int batchsize, Train *train_blob,
    float *weights, float *adamAvgMom, float *adamAvgVel, float *adam_t,
    float *bias, float *adamAvgMombias, float *adamAvgVelbias) {
    n->_IDinLayer = nodeID;
    n->_type = type;
    n->_train = train_blob + nodeID * batchsize;
    n->_weights = weights;
    n->_adamAvgMom = adamAvgMom;
    n->_adamAvgVel = adamAvgVel;
    n->_t = adam_t;
    n->_bias = bias;
    n->_adamAvgMombias = adamAvgMombias;
    n->_adamAvgVelbias = adamAvgVelbias;
}

float node_get_last_activation(Node *n, int inputID) {
    return (n->_train[inputID]._ActiveinputIds != 1) ? 0.0 : n->_train[inputID]._lastActivations;
}

void node_set_last_activation(Node *n, int inputID, float realActivation) {
    n->_train[inputID]._lastActivations = realActivation;
}

void node_increment_delta(Node *n, int inputID, float incrementValue) {
    assert(n->_train[inputID]._ActiveinputIds == 1);
    if (n->_train[inputID]._lastActivations > 0) n->_train[inputID]._lastDeltaforBPs += incrementValue;
}

float node_get_activation(Node *n, int *indices, float *values, int length, int inputID) {
    n->_train[inputID]._lastActivations = 0;
    for (int i = 0; i < length; i++) {
        n->_train[inputID]._lastActivations += n->_weights[indices[i]] * values[i];
    }
    n->_train[inputID]._lastActivations += *n->_bias;

    if (n->_train[inputID]._ActiveinputIds != 1) n->_train[inputID]._ActiveinputIds = 1; 

    switch (n->_type) {
    case ReLU:
        if (n->_train[inputID]._lastActivations < 0) {
            n->_train[inputID]._lastActivations = 0;
            n->_train[inputID]._lastDeltaforBPs = 0;  /* node_increment_delta will update this */
        }
        break;
    case Softmax:
        break;
    default:
        fprintf(stderr, "Invalid Node type\n");
        break;
    }
    return n->_train[inputID]._lastActivations;
}

bool node_get_input_active(Node *n, int inputID) {
    return n->_train[inputID]._ActiveinputIds == 1;
}

bool ID_in_label(int *label, int labelsize, int idd) {
  for (int i = 0; i < labelsize; i++) {
      if (label[i] == idd) return true;
  }
  return false;
}

void node_compute_softmax_stats(Node *n, float normalizationConstant, int inputID, int batchsize, int *label, int labelsize) {
    assert(n->_train[inputID]._ActiveinputIds == 1);
    n->_train[inputID]._lastActivations /= normalizationConstant + EPS;
    if (ID_in_label (label, labelsize, n->_IDinLayer)) {
        n->_train[inputID]._lastDeltaforBPs = (1.0/labelsize - n->_train[inputID]._lastActivations) / batchsize;
    }
    else {
        n->_train[inputID]._lastDeltaforBPs = (-n->_train[inputID]._lastActivations) / batchsize;
    }
}

/* can be done in parallel across inputs of a batch, provided layer prop is sequential */
void node_backprop(Node *n, Node *prevLayerNodeArray, int *prevLayerActiveNodeIds, int prevLayerActiveNodeSize,
    float learningRate, int inputID) {
    assert(n->_train[inputID]._ActiveinputIds == 1);
    for (int i = 0; i < prevLayerActiveNodeSize; i++) {
        // Update delta before updating weights
        Node *prev_node = &(prevLayerNodeArray[prevLayerActiveNodeIds[i]]);
        node_increment_delta(prev_node, inputID,
            n->_train[inputID]._lastDeltaforBPs * n->_weights[prevLayerActiveNodeIds[i]]);
        float grad_t = n->_train[inputID]._lastDeltaforBPs * node_get_last_activation(prev_node, inputID);
        n->_t[prevLayerActiveNodeIds[i]] += grad_t;
    }
}

/* can be done in parallel across inputs of a batch, provided layer prop is sequential */
void node_backprop_firstlayer(Node *n, int *nnzindices, float *nnzvalues, int nnzSize,
    float learningRate, int inputID) {
    assert(n->_train[inputID]._ActiveinputIds == 1);
    for (int i = 0; i < nnzSize; i++) {
        float grad_t = n->_train[inputID]._lastDeltaforBPs * nnzvalues[i];
        n->_t[nnzindices[i]] += grad_t;
    }
}

/* can be done at end of each batch, in parallel across nodes of a layer */
/* no other thread can access node's _weights, _bias, _train, _adamAvgVel, _adamAvgMom, -t */
/* including at the layer level above */
void node_adam(Node *n, int dim, int batchsize, float tmplr, int ratio) {
    float tbias = 0.0;

    for (int inputID=0; inputID<batchsize; inputID++){
        tbias += n->_train[inputID]._lastDeltaforBPs;
        n->_train[inputID]._lastDeltaforBPs = 0;
        n->_train[inputID]._lastActivations = 0;
        n->_train[inputID]._ActiveinputIds = 0;
    }

    for (int d=0; d<dim; d++){
        n->_adamAvgMom[d] = BETA1 * n->_adamAvgMom[d] + (1 - BETA1) * n->_t[d];
        n->_adamAvgVel[d] = BETA2 * n->_adamAvgVel[d] + (1 - BETA2) * n->_t[d] * n->_t[d];
        n->_weights[d]   += ratio * tmplr * n->_adamAvgMom[d] / (sqrt(n->_adamAvgVel[d]) + EPS);
        n->_t[d] = 0;
    }

    *n->_adamAvgMombias = BETA1 * (*n->_adamAvgMombias) + (1 - BETA1) * tbias;
    *n->_adamAvgVelbias = BETA2 * (*n->_adamAvgVelbias) + (1 - BETA2) * tbias * tbias;
    *n->_bias          += ratio * tmplr * (*n->_adamAvgMombias) / (sqrt(*n->_adamAvgVelbias) + EPS);
}

