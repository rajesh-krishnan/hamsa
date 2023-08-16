#include "hdefs.h"

void node_update(Node *n, int nodeID, NodeType type, int batchsize, Train *train_blob,
    float *weights, float *adamAvgMom, float *adamAvgVel, float *adam_t,
    float *bias, float *tbias, float *adamAvgMombias, float *adamAvgVelbias) {
    n->_IDinLayer = nodeID;
    n->_type = type;
    n->_train = train_blob + nodeID * batchsize;
    n->_weights = weights;
    n->_adamAvgMom = adamAvgMom;
    n->_adamAvgVel = adamAvgVel;
    n->_t = adam_t;
    n->_bias = bias;
    n->_tbias = tbias;
    n->_adamAvgMombias = adamAvgMombias;
    n->_adamAvgVelbias = adamAvgVelbias;
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
            n->_train[inputID]._lastDeltaforBPs = 0;
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

void node_backprop(Node *n, Node *prevLayerNodeArray, int *prevLayerActiveNodeIds, int prevLayerActiveNodeSize,
    float learningRate, int inputID) {
    assert(n->_train[inputID]._ActiveinputIds == 1);
    for (int i = 0; i < prevLayerActiveNodeSize; i++) {
        // Update delta before updating weights
        Node *prev_node = &(prevLayerNodeArray[prevLayerActiveNodeIds[i]]);
        node_increment_delta(prev_node, inputID,
            n->_train[inputID]._lastDeltaforBPs * n->_weights[prevLayerActiveNodeIds[i]]);
        float grad_t = n->_train[inputID]._lastDeltaforBPs * node_get_last_activation(prev_node, inputID);
#pragma omp atomic
        n->_t[prevLayerActiveNodeIds[i]] += grad_t;      /* _t is not per inputID, hence critical */
    }

#pragma omp atomic
    *n->_tbias += n->_train[inputID]._lastDeltaforBPs;   /* _tbias is not per inputID, hence atomic */

    n->_train[inputID]._lastDeltaforBPs = 0;
    n->_train[inputID]._lastActivations = 0;
    n->_train[inputID]._ActiveinputIds = 0;
}

void node_backprop_firstlayer(Node *n, int *nnzindices, float *nnzvalues, int nnzSize,
    float learningRate, int inputID) {
    assert(n->_train[inputID]._ActiveinputIds == 1);
    for (int i = 0; i < nnzSize; i++) {
        float grad_t = n->_train[inputID]._lastDeltaforBPs * nnzvalues[i];
#pragma omp atomic
        n->_t[nnzindices[i]] += grad_t;                  /* _t is not per inputID, hence atomic */
    }

#pragma omp atomic
    *n->_tbias += n->_train[inputID]._lastDeltaforBPs;   /* _tbias is not per inputID, hence atomic */

    n->_train[inputID]._lastDeltaforBPs = 0;
    n->_train[inputID]._lastActivations = 0;
    n->_train[inputID]._ActiveinputIds = 0;
}

/* done at end of each batch in parallel across nodes of a layer, not in parallel across inputs in a batch */
void node_adam(Node *n, int dim, int batchsize, float tmplr) {
#pragma omp simd
    for (int d=0; d<dim; d++) {
        n->_adamAvgMom[d] = BETA1 * n->_adamAvgMom[d] + (1 - BETA1) * n->_t[d];
        n->_adamAvgVel[d] = BETA2 * n->_adamAvgVel[d] + (1 - BETA2) * n->_t[d] * n->_t[d];
        n->_weights[d]   += tmplr * n->_adamAvgMom[d] / (sqrtf(n->_adamAvgVel[d]) + EPS);
        n->_t[d] = 0;
    }

    *n->_adamAvgMombias = BETA1 * (*n->_adamAvgMombias) + (1 - BETA1) * (*n->_tbias);
    *n->_adamAvgVelbias = BETA2 * (*n->_adamAvgVelbias) + (1 - BETA2) * (*n->_tbias) * (*n->_tbias);
    *n->_bias          += tmplr * (*n->_adamAvgMombias) / (sqrtf(*n->_adamAvgVelbias) + EPS);
    *n->_tbias          = 0.0;
}
