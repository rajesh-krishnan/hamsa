#include "node.h"

void node_update(Node *n, int dim, int nodeID, int layerID, NodeType type, int batchsize, 
    float *weights, float bias, float *adamAvgMom, float *adamAvgVel, float *adam_t, 
    Train* train_blob) {
    n->_dim = dim;
    n->_IDinLayer = nodeID;
    n->_layerNum = layerID;
    n->_type = type;
    n->_currentBatchsize = batchsize;
    n->_weights = weights;
    n->_bias = bias;

    n->_adamAvgMom = adamAvgMom;
    n->_adamAvgVel = adamAvgVel;
    n->_t = adam_t;

    n->_train = train_blob + nodeID * batchsize;

    n->_activeInputs = 0;
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

float node_get_activation(Node *n, int* indices, float* values, int length, int inputID) {
    assert(inputID <= n->_currentBatchsize);

    /* FUTURE TODO: shrink batchsize and check if input is already active; if so ignore backprop too */
    if (n->_train[inputID]._ActiveinputIds != 1) {
        n->_train[inputID]._ActiveinputIds = 1; //activate input
        n->_activeInputs++;
    }

    n->_train[inputID]._lastActivations = 0;
    for (int i = 0; i < length; i++) {
        n->_train[inputID]._lastActivations += n->_weights[indices[i]] * values[i];
    }
    n->_train[inputID]._lastActivations += n->_bias;

    switch (n->_type) {
    case ReLU:
        if (n->_train[inputID]._lastActivations < 0) {
            n->_train[inputID]._lastActivations = 0;
            n->_train[inputID]._lastGradients = 1;
            n->_train[inputID]._lastDeltaforBPs = 0;
        }else{
            n->_train[inputID]._lastGradients = 0;
        }
        break;
    case Softmax:
        break;
    default:
        fprintf(stderr, "Invalid Node type from Constructor\n");
        break;
    }
    return n->_train[inputID]._lastActivations;
}

bool node_get_input_active(Node *n, int inputID) {
    return n->_train[inputID]._ActiveinputIds == 1;
}

bool node_get_active_inputs(Node *n) {
    return n->_activeInputs > 0;
}

float node_perturb_weight(Node *n, int weightid, float delta) {
    n->_weights[weightid] += delta;
    return n->_weights[weightid];
}

float node_get_gradient(Node *n, int weightid, int inputID, float InputVal) {
    return -n->_train[inputID]._lastDeltaforBPs * InputVal;
}

bool ID_in_label(int * label, int labelsize, int idd) {
  for (int i = 0; i < labelsize; i++) {
      if (label[i] == idd) return true; 
  }
  return false;
}

void node_compute_softmax_stats(Node *n, float normalizationConstant, int inputID, int* label, int labelsize) {
    assert(n->_train[inputID]._ActiveinputIds == 1);

    n->_train[inputID]._lastActivations /= normalizationConstant + 0.0000001;

    /* TODO: check gradient */
    n->_train[inputID]._lastGradients = 1;
    if (ID_in_label (label, labelsize, n->_IDinLayer)) {
        n->_train[inputID]._lastDeltaforBPs = (1.0/labelsize - n->_train[inputID]._lastActivations) / n->_currentBatchsize;
    }
    else {
        n->_train[inputID]._lastDeltaforBPs = (-n->_train[inputID]._lastActivations) / n->_currentBatchsize;
    }
}

void node_backprop(Node *n, Node* previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, 
    float learningRate, int inputID) {
    assert(n->_train[inputID]._ActiveinputIds == 1);
    for (int i = 0; i < previousLayerActiveNodeSize; i++)
    {
        // Update delta before updating weights
        Node* prev_node = &(previousNodes[previousLayerActiveNodeIds[i]]);
        node_increment_delta(prev_node, inputID, 
            n->_train[inputID]._lastDeltaforBPs * n->_weights[previousLayerActiveNodeIds[i]]);
        float grad_t = n->_train[inputID]._lastDeltaforBPs * node_get_last_activation(prev_node, inputID);
        n->_t[previousLayerActiveNodeIds[i]] += grad_t;
    }
    n->_tbias += n->_train[inputID]._lastDeltaforBPs;
    n->_train[inputID]._ActiveinputIds = 0;
    n->_train[inputID]._lastDeltaforBPs = 0;
    n->_train[inputID]._lastActivations = 0;
    n->_activeInputs--;
}

void node_backprop_firstlayer(Node *n, int* nnzindices, float* nnzvalues, int nnzSize, 
    float learningRate, int inputID) {
    assert(n->_train[inputID]._ActiveinputIds == 1);
    for (int i = 0; i < nnzSize; i++) {
        float grad_t = n->_train[inputID]._lastDeltaforBPs * nnzvalues[i];
        n->_t[nnzindices[i]] += grad_t;
    }
    n->_tbias += n->_train[inputID]._lastDeltaforBPs;
    n->_train[inputID]._ActiveinputIds = 0;
    n->_train[inputID]._lastDeltaforBPs = 0;
    n->_train[inputID]._lastActivations = 0;
    n->_activeInputs--;
}

