#include "layer.h"

Layer *layer_new(size_t noOfNodes, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, 
    int K, int L, int RangePow, float Sparsity, float qSparsity, bool load, char * path) {
    Layer *l = mymap(sizeof(Layer));
    l->_noOfNodes = noOfNodes;
    l->_previousLayerNumOfNodes = previousLayerNumOfNodes;
    l->_layerID = layerID;
    l->_type = type;
    l->_batchsize = batchsize;
    l->_K = K;
    l->_L = L;
    l->_RangePow = RangePow;
    l->_Sparsity = Sparsity;
    l->_qSparsity = qSparsity;
    l->_noOfActive = floor(noOfNodes * Sparsity);

    l->_Nodes = mymap(noOfNodes * sizeof(Node));
    l->_hashTables = lsh_new(K, L, RangePow);
    l->_dwtaHasher = dwtahash_new(K * L, previousLayerNumOfNodes);
    l->_train_array = mymap(noOfNodes * batchsize * sizeof(Train));

    l->_randNode = mymap(noOfNodes * sizeof(int));
    for (size_t n = 0; n < noOfNodes; n++) l->_randNode[n] = n;
    myshuffle(l->_randNode, noOfNodes);

    size_t fano = noOfNodes * previousLayerNumOfNodes;
    l->_weights = mymap(fano * sizeof(float));
    l->_bias = mymap(noOfNodes * sizeof(float));
    l->_adamAvgMom = mymap(fano * sizeof(float));
    l->_adamAvgVel = mymap(fano * sizeof(float));
    l->_adamT = mymap(fano * sizeof(float));
    if (type == Softmax) l->_normalizationConstants = mymap(batchsize * sizeof(float));

    if (LOADWEIGHT) {
/*
        float *weight, *bias, *adamAvgMom, *adamAvgVel;
        if(LOADWEIGHT){
            cnpy::NpyArray weightArr, biasArr, adamArr, adamvArr;
            weightArr = arr["w_layer_"+to_string(i)];
            weight = weightArr.data<float>();
            biasArr = arr["b_layer_"+to_string(i)];
            bias = biasArr.data<float>();

            adamArr = arr["am_layer_"+to_string(i)];
            adamAvgMom = adamArr.data<float>();
            adamvArr = arr["av_layer_"+to_string(i)];
            adamAvgVel = adamvArr.data<float>();
        }
        _weights = weights;
        _bias = bias;
        _adamAvgMom = adamAvgMom;
        _adamAvgVel = adamAvgVel;
*/
    }else{

        random_device rd;
        default_random_engine dre(rd());
        normal_distribution<float> distribution(0.0, 0.01);

        generate(_weights, _weights + _noOfNodes * previousLayerNumOfNodes, [&] () { return distribution(dre); });
        generate(_bias, _bias + _noOfNodes, [&] () { return distribution(dre); });
    }

#pragma omp parallel for
    for (size_t i = 0; i < noOfNodes; i++)
    {
// will need dim and adam_t
        _Nodes[i].Update(previousLayerNumOfNodes, i, _layerID, type, batchsize, _weights+previousLayerNumOfNodes*i,
                _bias[i], _adamAvgMom+previousLayerNumOfNodes*i , _adamAvgVel+previousLayerNumOfNodes*i, _train_array);
        addtoHashTable(_Nodes[i]._weights, previousLayerNumOfNodes, _Nodes[i]._bias, i);
    }

}

void saveWeights(Layer *l, char *path) {
    if (_layerID==0) {
        cnpy::npz_save(file, "w_layer_0", _weights, {_noOfNodes, _Nodes[0]._dim}, "w");
        cnpy::npz_save(file, "b_layer_0", _bias, {_noOfNodes}, "a");
        cnpy::npz_save(file, "am_layer_0", _adamAvgMom, {_noOfNodes, _Nodes[0]._dim}, "a");
        cnpy::npz_save(file, "av_layer_0", _adamAvgVel, {_noOfNodes, _Nodes[0]._dim}, "a");
        cout<<"save for layer 0"<<endl;
        cout<<_weights[0]<<" "<<_weights[1]<<endl;
    }else{
        cnpy::npz_save(file, "w_layer_"+ to_string(_layerID), _weights, {_noOfNodes, _Nodes[0]._dim}, "a");
        cnpy::npz_save(file, "b_layer_"+ to_string(_layerID), _bias, {_noOfNodes}, "a");
        cnpy::npz_save(file, "am_layer_"+ to_string(_layerID), _adamAvgMom, {_noOfNodes, _Nodes[0]._dim}, "a");
        cnpy::npz_save(file, "av_layer_"+ to_string(_layerID), _adamAvgVel, {_noOfNodes, _Nodes[0]._dim}, "a");
        cout<<"save for layer "<<to_string(_layerID)<<endl;
        cout<<_weights[0]<<" "<<_weights[1]<<endl;
    }
}


void layer_delete(Layer *l) {
    size_t fano = l->_noOfNodes * l->_previousLayerNumOfNodes;
    lsh_delete(l->_hashTables);
    dwtahash_delete(l->_dwtaHasher);

    myunmap(l->_Nodes, l->_noOfNodes * sizeof(Node));
    myunmap(l->_train_array, l->_noOfNodes * l->_batchsize * sizeof(Train));
    myunmap(l->_randNode, l->_noOfNodes * sizeof(int));
    myunmap(l->_weights, fano * sizeof(float));
    myunmap(l->_bias, l->_noOfNodes * sizeof(float));
    myunmap(l->_adamAvgMom, fano * sizeof(float));
    myunmap(l->_adamAvgVel, fano * sizeof(float));
    myunmap(l->_adamT, fano * sizeof(float));
    if (type == Softmax) myunmap(l->_normalizationConstants, l->batchsize * sizeof(float));
}

void updateTable(Layer *l) {
    dwtahash_delete(l->_dwtaHasher);
    l->_dwtaHasher = dwtahash_new(l->_K * l->_L, l->_previousLayerNumOfNodes);
}

void updateRandomNodes(Layer *l) { myshuffle(l->_randNode, l->_noOfNodes); }

Node* getAllNodes(Layer *l) { return l->_Nodes; }

int getNodeCount(Layer *l) { return l->_noOfNodes; }

Node* getNodebyID(Layer *l, size_t nodeID) {
    assert(nodeID < l->_noOfNodes);
    return &l->_Nodes[nodeID];
}

float getNomalizationConstant(Layer *l, int inputID) {
    assert(l->_type == Softmax);
    return l->_normalizationConstants[inputID];
}

void Layer::addtoHashTable(float* weights, int length, float bias, int ID) {
    int *hashes = _dwtaHasher->getHashEasy(weights, length, TOPK);
    int *hashIndices = _hashTables->hashesToIndex(hashes);
    int *bucketIndices = _hashTables->add(hashIndices, ID+1);
    _Nodes[ID]._indicesInTables = hashIndices;
    _Nodes[ID]._indicesInBuckets = bucketIndices;
    delete [] hashes;
}

int Layer::queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* lengths, int layerIndex, int inputID, int* label, int labelsize, float Sparsity, int iter) {
    //LSH QueryLogic

    //Beidi. Query out all the candidate nodes
    int len;
    int in = 0;

    if(Sparsity == 1.0){
        len = _noOfNodes;
        lengths[layerIndex + 1] = len;
        activenodesperlayer[layerIndex + 1] = new int[len]; //assuming not intitialized;
        for (int i = 0; i < len; i++)
        {
            activenodesperlayer[layerIndex + 1][i] = i;
        }
    }
    else
    {
            int *hashes = _dwtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                              lengths[layerIndex]);
            int *hashIndices = _hashTables->hashesToIndex(hashes);
            int **actives = _hashTables->retrieveRaw(hashIndices);
            // we now have a sparse array of indices of active nodes

            // Get candidates from hashtable
            std::map<int, size_t> counts;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax && labelsize > 0) {
                for (int i = 0; i < labelsize ;i++){
                    counts[label[i]] = _L;
                }
            }

            for (int i = 0; i < _L; i++) {
                // copy sparse array into (dense) map
                for (int j = 0; j < BUCKETSIZE; j++) {
                    int tempID = actives[i][j] - 1;
                    if (tempID < 0) break;
                    counts[tempID] += 1;
                }
            }

            in = counts.size();
            if (counts.size()<1500){
                srand(time(NULL));
                size_t start = rand() % _noOfNodes;
                for (size_t i = start; i < _noOfNodes; i++) {
                    if (counts.size() >= 1000) {
                        break;
                    }
                    if (counts.count(_randNode[i]) == 0) {
                        counts[_randNode[i]] = 0;
                    }
                }

                if (counts.size() < 1000) {
                    for (size_t i = 0; i < _noOfNodes; i++) {
                        if (counts.size() >= 1000) {
                            break;
                        }
                        if (counts.count(_randNode[i]) == 0) {
                            counts[_randNode[i]] = 0;
                        }
                    }
                }
            }

            len = counts.size();
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            // copy map into new array
            int i=0;
            for (auto &&x : counts) {
                activenodesperlayer[layerIndex + 1][i] = x.first;
                i++;
            }

            delete[] hashes;
            delete[] hashIndices;
            delete[] actives;

    }

    //***********************************
    activeValuesperlayer[layerIndex + 1] = new float[len]; //assuming its not initialized else memory leak;
    float maxValue = 0;
    if (_type == NodeType::Softmax)
        _normalizationConstants[inputID] = 0;

    // find activation for all ACTIVE nodes in layer
    for (int i = 0; i < len; i++)
    {
        activeValuesperlayer[layerIndex + 1][i] = _Nodes[activenodesperlayer[layerIndex + 1][i]].getActivation(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex], inputID);
        if(_type == NodeType::Softmax && activeValuesperlayer[layerIndex + 1][i] > maxValue){
            maxValue = activeValuesperlayer[layerIndex + 1][i];
        }
    }

    if(_type == NodeType::Softmax) {
        for (int i = 0; i < len; i++) {
            float realActivation = exp(activeValuesperlayer[layerIndex + 1][i] - maxValue);
            activeValuesperlayer[layerIndex + 1][i] = realActivation;
            _Nodes[activenodesperlayer[layerIndex + 1][i]].SetlastActivation(inputID, realActivation);
            _normalizationConstants[inputID] += realActivation;
        }
    }

    return in;
}

