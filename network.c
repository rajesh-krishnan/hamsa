#include "network.h"

Network *network_new(int *sizesOfLayers, NodeType *layersTypes, int noOfLayers, int batchSize, float lr, 
    int inputdim, int *K, int *L, int *RangePow, float *Sparsity, bool load, char *path) {
    Network *n = mymap(sizeof(Network));
    n->_hiddenlayers = mymap(noOfLayers * sizeof(Layer *));
    n->_numberOfLayers = noOfLayers;
    n->_learningRate = lr;
    n->_currentBatchSize = batchSize;

#pragma omp parallel for
    for (int i = 0; i < noOfLayers; i++) {
        int sizes = (i != 0) ?  sizesOfLayers[i - 1] : inputdim;
        n->_hiddenlayers[i] = layer_new(sizesOfLayers[i], sizes, i, layersTypes[i], batchSize,
            K[i], L[i], RangePow[i], Sparsity[i], Sparsity[noOfLayers+i] load, path);
    }
}

void network_delete(Network *n) {
    for (int i = 0; i < n->_numberOfLayers; i++) layer_delete(n->_hiddenlayers[i]);
    myunmap(n->_hiddenlayers, n->numberOfLayers * sizeof(Layer *));
    myunmap(n, sizeof(Network));
}

Layer *getLayer(Network *n, int LayerID) {
    assert(LayerID >= 0 && LayerID < _numberOfLayers);
    return n->_hiddenlayers[LayerID];
}

void network_save(Network *n, char *path) {
    for (int i = 0; i < n->_numberOfLayers; i++) n->_hiddenlayers[i]->layer_save(path);
}

void network_load(Network *n, char *path) {
    for (int i = 0; i < n->_numberOfLayers; i++) n->_hiddenlayers[i]->layer_load(path);
}

int network_infer(Network *n, int **inputIndices, float **inputValues, int *length, int **labels, int *labelsize) {
    int correctPred = 0;

#pragma omp parallel for reduction(+:correctPred)
    for (int i = 0; i < _currentBatchSize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
        int *sizes = new int[_numberOfLayers + 1]();

        activenodesperlayer[0] = inputIndices[i];
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = length[i];

        //inference
        for (int j = 0; j < _numberOfLayers; j++) {
            _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], 0,
                    _Sparsity[_numberOfLayers+j], -1); // XXX: second half of sparsity array used for inference?
                                                       // XXX: why not just record in layer?
        }

        //compute softmax
        int noOfClasses = sizes[_numberOfLayers];
        float max_act = -222222222;
        int predict_class = -1;
        for (int k = 0; k < noOfClasses; k++) {
            float cur_act = _hiddenlayers[_numberOfLayers - 1]->getNodebyID(activenodesperlayer[_numberOfLayers][k])->getLastActivation(i);
            if (max_act < cur_act) {
                max_act = cur_act;
                predict_class = activenodesperlayer[_numberOfLayers][k];
            }
        }

        if (std::find (labels[i], labels[i]+labelsize[i], predict_class)!= labels[i]+labelsize[i]) {
            correctPred++;
        }

        delete[] sizes;
        for (int j = 1; j < _numberOfLayers + 1; j++) {
            delete[] activenodesperlayer[j];
            delete[] activeValuesperlayer[j];
        }
        delete[] activenodesperlayer;
        delete[] activeValuesperlayer;
    }
    return correctPred;
}

int network_train(Network *n, int **inputIndices, float **inputValues, int *lengths, int **labels, int *labelsize, 
    int iter, bool rehash, bool rebuild) {

    float logloss = 0.0;
    int* avg_retrieval = new int[_numberOfLayers]();

    for (int j = 0; j < _numberOfLayers; j++)
        avg_retrieval[j] = 0;


    if(iter%6946==6945 ){ // XXX: hack?
        //_learningRate *= 0.5;
        _hiddenlayers[1]->updateRandomNodes();
    }
    float tmplr = _learningRate;
    tmplr = _learningRate * sqrt((1 - pow(BETA2, iter + 1))) / (1 - pow(BETA1, iter + 1));

    int*** activeNodesPerBatch = new int**[_currentBatchSize];
    float*** activeValuesPerBatch = new float**[_currentBatchSize];
    int** sizesPerBatch = new int*[_currentBatchSize];
#pragma omp parallel for
    for (int i = 0; i < _currentBatchSize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
        int *sizes = new int[_numberOfLayers + 1]();

        activeNodesPerBatch[i] = activenodesperlayer;
        activeValuesPerBatch[i] = activeValuesperlayer;
        sizesPerBatch[i] = sizes;

        activenodesperlayer[0] = inputIndices[i];  // inputs parsed from training data file
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = lengths[i];
        int in;
        //auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < _numberOfLayers; j++) {
            in = _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], labelsize[i],
                    _Sparsity[j], iter*_currentBatchSize+i);
            avg_retrieval[j] += in;
        }

        //Now backpropagate.
        // layers
        for (int j = _numberOfLayers - 1; j >= 0; j--) {
            Layer* layer = _hiddenlayers[j];
            Layer* prev_layer = _hiddenlayers[j - 1];
            // nodes
            for (int k = 0; k < sizesPerBatch[i][j + 1]; k++) {
                Node* node = layer->getNodebyID(activeNodesPerBatch[i][j + 1][k]);
                if (j == _numberOfLayers - 1) {
                    //TODO: Compute Extra stats: labels[i];
                    node->ComputeExtaStatsForSoftMax(layer->getNomalizationConstant(i), i, labels[i], labelsize[i]);
                }
                if (j != 0) {
                    node->backPropagate(prev_layer->getAllNodes(), activeNodesPerBatch[i][j], sizesPerBatch[i][j], tmplr, i);
                } else {
                    node->backPropagateFirstLayer(inputIndices[i], inputValues[i], lengths[i], tmplr, i);
                }
            }
        }
    }
    for (int i = 0; i < _currentBatchSize; i++) {
        //Free memory to avoid leaks
        delete[] sizesPerBatch[i];
        for (int j = 1; j < _numberOfLayers + 1; j++) {
            delete[] activeNodesPerBatch[i][j];
            delete[] activeValuesPerBatch[i][j];
        }
        delete[] activeNodesPerBatch[i];
        delete[] activeValuesPerBatch[i];
    }

    delete[] activeNodesPerBatch;
    delete[] activeValuesPerBatch;
    delete[] sizesPerBatch;


    auto t1 = std::chrono::high_resolution_clock::now();
    bool tmpRehash;
    bool tmpRebuild;

    for (int l=0; l < _numberOfLayers; l++) {
        tmpRehash = (rehash & _Sparsity[l] < 1) ? true : false;
        tmpRebuild = (rebuild & _Sparsity[l] < 1) ? true : false;
        if (tmpRehash) _hiddenlayers[l]->_hashTables->clear();
        if (tmpRebuild) _hiddenlayers[l]->updateTable();

        int ratio = 1;
#pragma omp parallel for
        for (size_t m = 0; m < _hiddenlayers[l]->_noOfNodes; m++)
        {
            Node *tmp = _hiddenlayers[l]->getNodebyID(m);
            int dim = tmp->_dim;
            float* local_weights = new float[dim];
            std::copy(tmp->_weights, tmp->_weights + dim, local_weights);

            for (int d=0; d < dim;d++){
                float _t = tmp->_t[d];
                float Mom = tmp->_adamAvgMom[d];
                float Vel = tmp->_adamAvgVel[d];
                Mom = BETA1 * Mom + (1 - BETA1) * _t;
                Vel = BETA2 * Vel + (1 - BETA2) * _t * _t;
                local_weights[d] += ratio * tmplr * Mom / (sqrt(Vel) + EPS);
                tmp->_adamAvgMom[d] = Mom;
                tmp->_adamAvgVel[d] = Vel;
                tmp->_t[d] = 0;
            }

            tmp->_adamAvgMombias = BETA1 * tmp->_adamAvgMombias + (1 - BETA1) * tmp->_tbias;
            tmp->_adamAvgVelbias = BETA2 * tmp->_adamAvgVelbias + (1 - BETA2) * tmp->_tbias * tmp->_tbias;
            tmp->_bias += ratio*tmplr * tmp->_adamAvgMombias / (sqrt(tmp->_adamAvgVelbias) + EPS);
            tmp->_tbias = 0;
            std::copy(local_weights, local_weights + dim, tmp->_weights);
            if (tmpRehash) {
                int *hashes = _hiddenlayers[l]->_dwtaHasher->getHashEasy(local_weights, dim, TOPK);
                int *hashIndices = _hiddenlayers[l]->_hashTables->hashesToIndex(hashes);
                int * bucketIndices = _hiddenlayers[l]->_hashTables->add(hashIndices, m+1);

                delete[] hashes;
                delete[] hashIndices;
                delete[] bucketIndices;
            }

            delete[] local_weights;
        }
    }

    if (DEBUG&rehash) {
        cout << "Avg sample size = " << avg_retrieval[0]*1.0/_currentBatchSize<<" "<<avg_retrieval[1]*1.0/_currentBatchSize << endl;
    }
    return logloss;
}

