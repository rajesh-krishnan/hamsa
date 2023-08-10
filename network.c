#include "hdefs.h"

Network *network_new(Config *cfg, bool loadParams) {
    Network *n = (Network *) mymap(sizeof(Network));
    n->_cfg = cfg;
    n->_hiddenlayers = (Layer **) mymap(cfg->numLayer * sizeof(Layer *));
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
    myunmap(n->_hiddenlayers, n->_cfg->numLayer * sizeof(Layer *));
    myunmap(n, sizeof(Network));
}

void network_save_params(Network *n) {
    for (int i = 0; i < n->_cfg->numLayer; i++) layer_save(n->_hiddenlayers[i], n->_cfg->savePath);
}

void network_load_params(Network *n) {
    for (int i = 0; i < n->_cfg->numLayer; i++) layer_load(n->_hiddenlayers[i], n->_cfg->loadPath);
}

/* Reuse code block in training and inference */
#define ALLOC_ACTIVEOUT_FWDPROP( S ) \
            int lengthIn, lengthOut=0; \
            int *activeNodesIn, *activeNodesOut; \
            float *activeValuesIn, *activeValuesOut; \
            activeNodesIn   = (j==0) ? inIndices[i] : activeNodesOut; \
            activeValuesIn  = (j==0) ? inValues[i]  : activeValuesOut; \
            lengthIn        = (j==0) ? inLength[i]  : lengthOut; \
            activeNodesOut  = (int *)   malloc(n->_hiddenlayers[j]->_noOfNodes * sizeof(int)); \
            activeValuesOut = (float *) malloc(n->_hiddenlayers[j]->_noOfNodes * sizeof(int)); \
            assert((activeNodesOut != NULL) && (activeValuesOut != NULL)); \
            int avr = layer_fwdprop(n->_hiddenlayers[j], activeNodesIn, activeValuesIn, lengthIn, \
                activeNodesOut, activeValuesOut, &lengthOut, i, blabels[i], blabelsize[i], Sparsity);

void network_train(Network *n, int **inIndices, float **inValues, int *inLength, int **blabels, int *blabelsize,
    int iter, bool reperm, bool rehash, bool rebuild) {
    int *avg_retrieval = (int *) malloc(n->_cfg->numLayer * n->_cfg->Batchsize * sizeof(int));
    assert(avg_retrieval != NULL);
    memset(avg_retrieval, 0, n->_cfg->numLayer * n->_cfg->Batchsize * sizeof(int)); /* init to 0 */
    float tmplr = n->_cfg->Lr * sqrt((1 - pow(BETA2, iter + 1))) / (1 - pow(BETA1, iter + 1));

#pragma omp parallel for
    for (int i = 0; i < n->_cfg->Batchsize; i++) {
        int   **activeNodes  = (int **)   malloc(n->_cfg->numLayer * sizeof(int *));
        float **activeValues = (float **) malloc(n->_cfg->numLayer * sizeof(float *));
        int   *activeLength  = (int *)    malloc(n->_cfg->numLayer * sizeof(int));
        for (int j = 0; j < n->_cfg->numLayer; j++) {                 /* forward prop across layers */
            float Sparsity  = n->_cfg->Sparsity[j];                   /* use second half for train */
            ALLOC_ACTIVEOUT_FWDPROP( Sparsity );
            avg_retrieval[i * n->_cfg->numLayer + j] += avr;          /* save stats */
            activeNodes[j]  = activeNodesOut;                         /* save for backprop */
            activeValues[j] = activeValuesOut;
            activeLength[j] = lengthOut;
        }

        for (int j = n->_cfg->numLayer - 1; j >= 0; j--) {           /* back prop across layers */
            Layer* thisLay = n->_hiddenlayers[j];
            Layer* prevLay = n->_hiddenlayers[j-1];
            if (j == n->_cfg->numLayer - 1) {
                layer_compute_softmax_stats(thisLay, activeNodes[j], activeLength[j],
                     thisLay->_normalizationConstants[i], i, blabels[i], blabelsize[i]);
            }
            if (j != 0) {
                layer_backprop(thisLay, activeNodes[j], activeLength[j], prevLay,
                    activeNodes[j-1], activeLength[j-1], tmplr, i);
            }
            else {
                layer_backprop_firstlayer(thisLay, activeNodes[j], activeLength[j],
                    inIndices[i], inValues[i], inLength[i], tmplr, i);
            }
        }

        for(int k = 0; k < n->_cfg->numLayer; k++) {
            free(activeNodes[k]);
            free(activeValues[k]);
        }
        free(activeNodes);
        free(activeValues);
        free(activeLength);
    }

    // gradient descent after each batch
    for (int j=0; j < n->_cfg->numLayer; j++) {
        float Sparsity  = n->_cfg->Sparsity[j];                /* use first half for training */
        Layer *l        = n->_hiddenlayers[j];
        bool last       = (n->_cfg->numLayer - 1);
        layer_adam(l, tmplr, 1);
        if (rebuild && (Sparsity < 1)) layer_updateHasher(l);
        if (rehash && (Sparsity < 1))  layer_rehash(l);
        if (reperm && last)            layer_updateRandomNodes(l);   /* XXX: why only last layer */
        if (rehash) {
            int avg = 0;
            for (int i = 0; i < n->_cfg->Batchsize; i++) avg += avg_retrieval[i * n->_cfg->numLayer + j];
            fprintf(stderr, "Layer %d average sample size = %lf Sparsity=%f\n", j, avg*1.0/n->_cfg->Batchsize, Sparsity);
        }
    }
}

int network_infer(Network *n, int **inIndices, float **inValues, int *inLength, int **blabels, int *blabelsize) {
    int correctPred = 0;
#pragma omp parallel for reduction(+:correctPred)
    for (int i = 0; i < n->_cfg->Batchsize; i++) {
        int predict_class = -1;
        for (int j = 0; j < n->_cfg->numLayer; j++) {
            float Sparsity  = n->_cfg->Sparsity[n->_cfg->numLayer + j];  /* use second half for infer */
            ALLOC_ACTIVEOUT_FWDPROP( Sparsity );
            if (j != 0) { free(activeNodesIn);  free(activeValuesIn); } /* free the previous layer's actives */
            if (j == n->_cfg->numLayer) {                    /* get prediction and free last layer's actives */
                predict_class = layer_get_prediction(n->_hiddenlayers[j], activeNodesOut, lengthOut, i);
                free(activeNodesOut); free(activeValuesOut);
            }
        }
        for(int k=0; k < blabelsize[i]; k++)
            if(blabels[i][k] == predict_class) { correctPred += 1; break; }
    }
    return correctPred;
}

