#include "layer.h"

Layer *layer_new(size_t noOfNodes, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, 
    int K, int L, int RangePow, float Sparsity, float qSparsity, bool load, char *path) {
    Layer *l = mymap(sizeof(Layer));

    myrnginit();
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
    (load) ?  layer_load(l, path) : layer_randinit(l);

#pragma omp parallel for
    for (size_t i = 0; i < noOfNodes; i++)
    {
        size_t index = previousLayerNumOfNodes * i;
        node_update(&l->_Nodes[i], i, type, batchsize,
            &l->_weights[index], l->_bias[i], &l->_adamAvgMom[index], &l->_adamAvgVel[index], &l->_adamT[index], 
            l->_train_array);
        layer_addToHashTable(l, &l->_weights[index], previousLayerNumOfNodes, i);
    }
    return l;
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
    if (l->_type == Softmax) myunmap(l->_normalizationConstants, l->_batchsize * sizeof(float));
    myunmap(l, sizeof(Layer));
}

void layer_randinit(Layer *l) {
    size_t fano = l->_noOfNodes * l->_previousLayerNumOfNodes;
#pragma omp parallel for
    for (size_t i = 0; i < fano; i++) l->_weights[i] = randnorm(0.0,0.01);
#pragma omp parallel for
    for (size_t i = 0; i < l->_noOfNodes; i++) l->_bias[i] = randnorm(0.0,0.01);
}

static void layer_rw(Layer *l, char *path, bool read) {
    char fn[1024];
    size_t len = strlen(path);
    assert(len < 1000);
    strcpy(fn, path);
    void (*rwfn)(float*,bool,size_t,size_t,char*);
    rwfn = read ? read_fnpy : write_fnpy;
    sprintf(fn+len, "/b_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_bias, false, l->_noOfNodes, 1, fn);
    sprintf(fn+len, "/w_layer_%d.npy", l->_layerID);

    (*rwfn)(l->_weights, true, l->_noOfNodes, l->_previousLayerNumOfNodes, fn);
    sprintf(fn+len, "/am_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_adamAvgMom, true, l->_noOfNodes, l->_previousLayerNumOfNodes, fn);
    sprintf(fn+len, "/av_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_adamAvgVel, true, l->_noOfNodes, l->_previousLayerNumOfNodes, fn);
    fprintf(stderr, "%s bias, weights, moments, and velocities for layer %d\n", read ? "Loaded" : "Saved", l->_layerID);
}

void layer_load(Layer *l, char *path) { layer_rw(l, path, true); }

void layer_save(Layer *l, char *path) { layer_rw(l, path, false); }

Node *layer_getNodebyID(Layer *l, size_t nodeID) {
    assert((nodeID >= 0) && (nodeID < l->_noOfNodes));
    return &l->_Nodes[nodeID];
}

Node *layer_getAllNodes(Layer *l) { return l->_Nodes; }

int layer_getNodeCount(Layer *l) { return l->_noOfNodes; }

float layer_getNormalizationConstant(Layer *l, int inputID) {
    assert(l->_type == Softmax);
    return l->_normalizationConstants[inputID];
}

void layer_updateTable(Layer *l) {
    dwtahash_delete(l->_dwtaHasher);
    l->_dwtaHasher = dwtahash_new(l->_K * l->_L, l->_previousLayerNumOfNodes);
}

void layer_updateRandomNodes(Layer *l) { myshuffle(l->_randNode, l->_noOfNodes); }

void layer_addToHashTable(Layer *l, float* weights, int length, int id) {
    int *hashes = dwtahash_getHashEasy(l->_dwtaHasher, weights, length);
    lsh_hashes_to_indices_add(l->_hashTables, hashes, id + 1);
    free(hashes);
}

#if 0
int layer_queryActiveNodeandComputeActivations(Layer *l, int **activenodesperlayer, float **activeValuesperlayer, 
    int *lengths, int layerIndex, int inputID,  int *label, int labelsize, float Sparsity, int iter) {

// replace sparsity arg with tORq?
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

#endif
