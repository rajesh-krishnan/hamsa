#include "node.h"
#include "hdefs.h"

void Update(Node *n, int dim, int nodeID, int layerID, NodeType type, int batchsize, 
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
    n->_mirrorbias = bias;
}

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

/*
float Node::getLastActivation(int inputID)
{
    if(_train[inputID]._ActiveinputIds != 1)
        return 0.0;
    return _train[inputID]._lastActivations;
}


void Node::incrementDelta(int inputID, float incrementValue)
{
    assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
    if (_train[inputID]._lastActivations > 0)
        _train[inputID]._lastDeltaforBPs += incrementValue;
}

bool Node::getInputActive(int inputID)
{
    return _train[inputID]._ActiveinputIds == 1;
}

bool Node::getActiveInputs(void)
{
    return _activeInputs > 0;
}

float Node::getActivation(int* indices, float* values, int length, int inputID)
{
    assert(("Input ID more than Batch Size", inputID <= _currentBatchsize));

    //FUTURE TODO: shrink batchsize and check if input is alread active then ignore and ensure backpopagation is ignored too.
    if (_train[inputID]._ActiveinputIds != 1) {
        _train[inputID]._ActiveinputIds = 1; //activate input
        _activeInputs++;
    }

    _train[inputID]._lastActivations = 0;
    for (int i = 0; i < length; i++)
    {
        _train[inputID]._lastActivations += _weights[indices[i]] * values[i];
    }
    _train[inputID]._lastActivations += _bias;

    switch (_type)
    {
    case NodeType::ReLU:
        if (_train[inputID]._lastActivations < 0) {
            _train[inputID]._lastActivations = 0;
            _train[inputID]._lastGradients = 1;
            _train[inputID]._lastDeltaforBPs = 0;

        }else{
            _train[inputID]._lastGradients = 0;
        }
        break;
    case NodeType::Softmax:

        break;
    default:
        cout << "Invalid Node type from Constructor" <<endl;
        break;
    }

    return _train[inputID]._lastActivations;
}


void Node::ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize)
{
    assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds ==1));

    _train[inputID]._lastActivations /= normalizationConstant + 0.0000001;

    //TODO:check  gradient
    _train[inputID]._lastGradients = 1;
    if (find (label, label+labelsize, _IDinLayer)!= label+labelsize) {
        _train[inputID]._lastDeltaforBPs = (1.0/labelsize - _train[inputID]._lastActivations) / _currentBatchsize;
    }
    else {
        _train[inputID]._lastDeltaforBPs = (-_train[inputID]._lastActivations) / _currentBatchsize;
    }
}


void Node::backPropagate(Node* previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID)
{
    assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
    for (int i = 0; i < previousLayerActiveNodeSize; i++)
    {
        //UpdateDelta before updating weights
        Node* prev_node = &(previousNodes[previousLayerActiveNodeIds[i]]);
        prev_node->incrementDelta(inputID, _train[inputID]._lastDeltaforBPs * _weights[previousLayerActiveNodeIds[i]]);

        float grad_t = _train[inputID]._lastDeltaforBPs * prev_node->getLastActivation(inputID);

        if (ADAM)
        {
            _t[previousLayerActiveNodeIds[i]] += grad_t;
        }
        else
        {
            _mirrorWeights[previousLayerActiveNodeIds[i]] += learningRate * grad_t;
        }
    }

    if (ADAM)
    {
        float biasgrad_t = _train[inputID]._lastDeltaforBPs;
        float biasgrad_tsq = biasgrad_t * biasgrad_t;
        _tbias += biasgrad_t;
    }
    else
    {
        _mirrorbias += learningRate * _train[inputID]._lastDeltaforBPs;
    }

    _train[inputID]._ActiveinputIds = 0;
    _train[inputID]._lastDeltaforBPs = 0;
    _train[inputID]._lastActivations = 0;
    _activeInputs--;

}


void Node::backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID)
{
    assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
    for (int i = 0; i < nnzSize; i++)
    {
        float grad_t = _train[inputID]._lastDeltaforBPs * nnzvalues[i];
        float grad_tsq = grad_t * grad_t;
        if (ADAM)
        {
            _t[nnzindices[i]] += grad_t;
        }
        else
        {
            _mirrorWeights[nnzindices[i]] += learningRate * grad_t;
        }
    }

    if (ADAM)
    {
        float biasgrad_t = _train[inputID]._lastDeltaforBPs;
        float biasgrad_tsq = biasgrad_t * biasgrad_t;
        _tbias += biasgrad_t;
    }
    else
    {
        _mirrorbias += learningRate * _train[inputID]._lastDeltaforBPs;
    }

    _train[inputID]._ActiveinputIds = 0;//deactivate inputIDs
    _train[inputID]._lastDeltaforBPs = 0;
    _train[inputID]._lastActivations = 0;
    _activeInputs--;
}

void Node::SetlastActivation(int inputID, float realActivation)
{
    _train[inputID]._lastActivations = realActivation;
}

Node::~Node()
{

    delete[] _indicesInTables;
    delete[] _indicesInBuckets;

    if (ADAM)
    {
        delete[] _adamAvgMom;
        delete[] _adamAvgVel;
        delete[] _t;
    }
}


// for debugging gradients.
float Node::purturbWeight(int weightid, float delta)
{
    _weights[weightid] += delta;
    return _weights[weightid];
}


float Node::getGradient(int weightid, int inputID, float InputVal)
{
    return -_train[inputID]._lastDeltaforBPs * InputVal;
}
*/
