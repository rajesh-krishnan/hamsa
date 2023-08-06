#include "network.h"

Network *network_new(Config *cfg, bool loadParams) {
    Network *n = (Network *) malloc(sizeof(Network));
    n->_cfg = cfg;
    n->_hiddenlayers = (Layer **) malloc(cfg->numLayer * sizeof(Layer *));
    assert((n != NULL) && (n->_hiddenlayers != NULL));
    
#pragma omp parallel for
    for (int i = 0; i < cfg->numLayer; i++) {
        int lsize = (i != 0) ?  cfg->sizesOfLayers[i - 1] : cfg->InputDim;
        n->_hiddenlayers[i] = layer_new(cfg->sizesOfLayers[i], lsize, i, cfg->layersTypes[i], cfg->Batchsize,
            cfg->K[i], cfg->L[i], cfg->RangePow[i], loadParams, cfg->loadPath);
    }
    return n;
}

void network_delete(Network *n) {
    for (int i = 0; i < n->_cfg->numLayer; i++) layer_delete(n->_hiddenlayers[i]);
    free(n->_hiddenlayers);
    free(n);
}

void network_save_params(Network *n) {
    for (int i = 0; i < n->_cfg->numLayer; i++) layer_save(n->_hiddenlayers[i], n->_cfg->savePath);
}

void network_load_params(Network *n) {
    for (int i = 0; i < n->_cfg->numLayer; i++) layer_load(n->_hiddenlayers[i], n->_cfg->loadPath);
}

#if 0
int network_infer(Network *n, int **inputIndices, float **inputValues, int *length, int **labels, int *labelsize) {
    int correctPred = 0;

#pragma omp parallel for reduction(+:correctPred)
    for (int i = 0; i < _currentBatchSize; i++) {
        int **activeNodes = new int *[_numberOfLayers + 1]();
        float **activeValues= new float *[_numberOfLayers + 1]();
        // allocate activeNodes and activeValues for relevant layers
        int *sizes = new int[_numberOfLayers + 1]();

        activeNodes[0] = inputIndices[i];
        activeValues[0] = inputValues[i];
        sizes[0] = length[i];

        // inference
        for (int j = 0; j < _numberOfLayers; j++) {
            layer_forwardPropagate(n->_hiddenlayers[j], 
                activeNodes, activeValues, 
                sizes, j, i, labels[i], 0, _Sparsity[_numberOfLayers+j], -1); // XXX: second half of sparsity arr?
        }

        //compute softmax
        int noOfClasses = sizes[_numberOfLayers];
        float max_act = -222222222;
        int predict_class = -1;
        for (int k = 0; k < noOfClasses; k++) {
            float cur_act = _hiddenlayers[_numberOfLayers - 1]->getNodebyID(activeNodes[_numberOfLayers][k])->getLastActivation(i);
            if (max_act < cur_act) {
                max_act = cur_act;
                predict_class = activeNodes[_numberOfLayers][k];
            }
        }

        if (std::find (labels[i], labels[i]+labelsize[i], predict_class)!= labels[i]+labelsize[i]) {
            correctPred++;
        }

        delete[] sizes;
        for (int j = 1; j < _numberOfLayers + 1; j++) {
            delete[] activeNodes[j];
            delete[] activeValues[j];
        }
        delete[] activeNodes;
        delete[] activeValues;
    }
    return correctPred;
}

void network_train(Network *n, int **inputIndices, float **inputValues, int *lengths, int **labels, int *labelsize, 
    int iter, bool rehash, bool rebuild) {

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
        int **activeNodes = new int *[_numberOfLayers + 1]();
        float **activeValues= new float *[_numberOfLayers + 1]();
        // allocate activeNodes and activeValues for relevant layers
        int *sizes = new int[_numberOfLayers + 1]();

        activeNodesPerBatch[i] = activeNodes ;
        activeValuesPerBatch[i] = activeValues;
        sizesPerBatch[i] = sizes;

        activeNodes [0] = inputIndices[i];  // inputs parsed from training data file
        activeValues[0] = inputValues[i];
        sizes[0] = lengths[i];

        // forward propagate
        int in;
        for (int j = 0; j < _numberOfLayers; j++) {
            in = layer_forwardPropagate(n->_hiddenlayers[j], activeNodes , activeValues, 
                sizes, j, i, labels[i], labelsize[i], _Sparsity[j], iter*_currentBatchSize+i);
            avg_retrieval[j] += in;
        }

        // back propagate 
        for (int j = _numberOfLayers - 1; j >= 0; j--) {
            Layer* layer = _hiddenlayers[j];
            Layer* prev_layer = _hiddenlayers[j - 1];
            for (int k = 0; k < sizesPerBatch[i][j + 1]; k++) {
                Node* node = layer->getNodebyID(activeNodesPerBatch[i][j + 1][k]);
                if (j == _numberOfLayers - 1) {
                    //TODO: Compute Extra stats: labels[i]; XXX: check LNS
                    node->ComputeExtaStatsForSoftMax(layer->_normalizationConstant[i], i, labels[i], labelsize[i]);
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

    // gradient descent
    bool tmpRehash;
    bool tmpRebuild;
    for (int l=0; l < _numberOfLayers; l++) {
        tmpRehash = (rehash & _Sparsity[l] < 1) ? true : false;
        tmpRebuild = (rebuild & _Sparsity[l] < 1) ? true : false;
        if (tmpRehash) _hiddenlayers[l]->_hashTables->clear();
        if (tmpRebuild) _hiddenlayers[l]->updateTable();
#pragma omp parallel for
        for (size_t m = 0; m < _hiddenlayers[l]->_noOfNodes; m++)
        {
            Node *tmp = _hiddenlayers[l]->getNodebyID(m);
            int dim = _hiddenlayers[l]->_prevLayerNumOfNodes;
            node_adam(tmp, dim, tmplr, 1);
            if (tmpRehash) layer_addToHashTable(n->_hiddenlayers[l], tmp->_weights, dim, m+1);
        }
    }

    if (DEBUG&rehash) {
        cout << "Avg sample size = " << avg_retrieval[0]*1.0/_currentBatchSize<<" "<<avg_retrieval[1]*1.0/_currentBatchSize << endl;
    }
}

#endif
