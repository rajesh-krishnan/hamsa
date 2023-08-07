#include "hdefs.h"

Network *network_new(Config *cfg, bool loadParams) {
    Network *n = (Network *) malloc(sizeof(Network));
    n->_cfg = cfg;
    n->_hiddenlayers = (Layer **) malloc(cfg->numLayer * sizeof(Layer *));
    assert((n != NULL) && (n->_hiddenlayers != NULL));
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

int network_infer(Network *n, int **inIndices, float **inValues, int *inLength, int **blabels, int *blabelsize) {
    int correctPred = 0;
#pragma omp parallel for reduction(+:correctPred)
    for (int i = 0; i < n->_cfg->Batchsize; i++) {
        int predict_class = -1; 
        for (int j = 0; j < n->_cfg->numLayer; j++) {
            int lengthIn, lengthOut=0;
            int *activeNodesIn, *activeNodesOut;
            float *activeValuesIn, *activeValuesOut;
            Layer *thisLay  = n->_hiddenlayers[j];
            int *label      = blabels[i];
            int labelsize   = blabelsize[i];
            float Sparsity  = n->_cfg->Sparsity[n->_cfg->numLayer + j];  /* use second half for infer */
            int maxlenOut   = thisLay->_noOfNodes;                       /* max active <= layer size */
            bool last       = (j == n->_cfg->numLayer);
            bool first      = (j == 0);

            activeNodesIn   = first ? inIndices[i] : activeNodesOut;
            activeValuesIn  = first ? inValues[i]  : activeValuesOut;
            lengthIn        = first ? inLength[i]  : lengthOut;
      
            activeNodesOut  = (int *)   malloc(maxlenOut * sizeof(int));   
            activeValuesOut = (float *) malloc(maxlenOut * sizeof(float));
            assert((activeNodesOut != NULL) && (activeValuesOut != NULL));

            layer_forwardPropagate(n->_hiddenlayers[j], activeNodesIn, activeValuesIn, lengthIn,
                activeNodesOut, activeValuesOut, &lengthOut, i, label, labelsize, Sparsity);

            if (!first) { free(activeNodesIn);  free(activeValuesIn); }
            if (last)   predict_class = layer_get_prediction(thisLay, activeNodesOut, lengthOut, i);
            if (last)   { free(activeNodesOut); free(activeValuesOut); break; }
        }
        for(int k=0; k < blabelsize[i]; k++) 
            if(blabels[i][k] == predict_class) { correctPred += 1; break; }
    }
    return correctPred;
}

void network_train(Network *n, int **inputIndices, float **inputValues, int *lengths, int **label, int *labelsize, 
    int iter, bool reperm, bool rehash, bool rebuild) {
    int numLayer = n->_cfg->numLayer;
    int *avg_retrieval = (int *) malloc(numLayer * sizeof(int));
    assert(avg_retrieval != NULL);
    for (int j = 0; j < numLayer; j++) avg_retrieval[j] = 0;

    float tmplr = n->_cfg->Lr * sqrt((1 - pow(BETA2, iter + 1))) / (1 - pow(BETA1, iter + 1));

// #pragma omp parallel for // check if safe for training
    for (int i = 0; i < n->_cfg->Batchsize; i++) {
#if 0
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
        for (int j = 0; j < _numberOfLayers; j++) {
            avg_retrieval[j] += layer_forwardPropagate(n->_hiddenlayers[j], activeNodes , activeValues, 
                sizes, j, i, labels[i], labelsize[i], _Sparsity[j], iter*_currentBatchSize+i);
        }

        // back propagate 
        for (int j = _numberOfLayers - 1; j >= 0; j--) {
            Layer* layer = _hiddenlayers[j];
            Layer* prev_layer = _hiddenlayers[j - 1];
            for (int k = 0; k < sizesPerBatch[i][j + 1]; k++) {
                Node* node = layer->getNodebyID(activeNodesPerBatch[i][j + 1][k]);
                if (j == _numberOfLayers - 1) {
                    node->ComputeExtaStatsForSoftMax(layer->_normalizationConstant[i], i, labels[i], labelsize[i]);
                }
                if (j != 0) {
                    node->backPropagate(prev_layer->getAllNodes(), activeNodesPerBatch[i][j], sizesPerBatch[i][j], tmplr, i);
                } else {
                    node->backPropagateFirstLayer(inputIndices[i], inputValues[i], lengths[i], tmplr, i);
                }
            }
        }
#endif
    }

    // XXX: deallocate memory

    // gradient descent after each batch
    for (int j=0; j < numLayer; j++) {
        float Sparsity  = n->_cfg->Sparsity[j];                /* use first half for training */
        Layer *l        = n->_hiddenlayers[l];
        bool last       = (numLayer - 1);
        layer_adam(l, tmplr, 1);
        if (rebuild && (Sparsity < 1)) layer_updateHasher(l);
        if (rehash && (Sparsity < 1))  layer_rehash(l);
        if (reperm && last)            layer_updateRandomNodes(l);
        if (rehash) 
             fprintf(stderr, "Layer %d average sample size = %lf\n", j, avg_retrieval[j]*1.0/n->_cfg->Batchsize);
    }
}
