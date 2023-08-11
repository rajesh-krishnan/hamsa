#include "hdefs.h"

inline static size_t __attribute__((always_inline)) layer_size(size_t noOfNodes, int prevLayerNumOfNodes, 
    int batchsize) {
    size_t hugepg_size = (2L << 21);  /* 2MB Hugepage */
    size_t fano = noOfNodes * prevLayerNumOfNodes;
    size_t buffer_size = sizeof(Layer) + noOfNodes * sizeof(Node) + noOfNodes * batchsize * sizeof(Train) + 
        4 * fano * sizeof(float) + 3 * noOfNodes * sizeof(float) + noOfNodes * sizeof(int) + batchsize * sizeof(float);
    return (size_t) ceil(hugepg_size * buffer_size * 1.0 / hugepg_size);
}

Layer *layer_new(size_t noOfNodes, int prevLayerNumOfNodes, int layerID, NodeType type, int batchsize, 
    int K, int L, int RangePow, bool load, char *path) {
    size_t fano = noOfNodes * prevLayerNumOfNodes;
    size_t buffer_size = layer_size(noOfNodes, prevLayerNumOfNodes, batchsize);
    void *buf = mymap(buffer_size);
    fprintf(stderr, "Allocated %ld bytes for Layer %d at %p\n", buffer_size, layerID, buf);

    Layer *l                   = (Layer *) buf; buf += sizeof(Layer);
    l->_Nodes                  = (Node *)  buf; buf += noOfNodes * sizeof(Node);
    l->_train_array            = (Train *) buf; buf += noOfNodes * batchsize * sizeof(Train);
    l->_weights                = (float *) buf; buf += fano * sizeof(float);
    l->_adamAvgMom             = (float *) buf; buf += fano * sizeof(float);
    l->_adamAvgVel             = (float *) buf; buf += fano * sizeof(float);
    l->_adamT                  = (float *) buf; buf += fano * sizeof(float);
    l->_bias                   = (float *) buf; buf += noOfNodes * sizeof(float);
    l->_adamAvgMombias         = (float *) buf; buf += noOfNodes * sizeof(float);
    l->_adamAvgVelbias         = (float *) buf; buf += noOfNodes * sizeof(float);
    l->_randNode               = (int *)   buf; buf += noOfNodes * sizeof(int);
    l->_normalizationConstants = (float *) buf; buf += batchsize * sizeof(float);
    l->_hashTables             = lsht_new(K, L, RangePow);
    l->_dwtaHasher             = dwtahash_new(K * L, prevLayerNumOfNodes);
    l->_noOfNodes              = noOfNodes;
    l->_prevLayerNumOfNodes    = prevLayerNumOfNodes;
    l->_layerID                = layerID;
    l->_type                   = type;
    l->_batchsize              = batchsize;
    l->_K                      = K;
    l->_L                      = L;
    l->_RangePow               = RangePow;

    for (size_t n = 0; n < noOfNodes; n++) l->_randNode[n] = n;      /* Init randNode array */
    layer_updateRandomNodes(l);                                      /* Shuffle randNode array */

    for (size_t k = 0; k < fano; k++) l->_adamT[k] = 0.0;            /* Set ADAM t array to 0 */

    size_t tblks = noOfNodes * batchsize;
    for (size_t k = 0; k < tblks; k++) {                             /* Init train array */
        l->_train_array[k]._lastDeltaforBPs = 0.0;
        l->_train_array[k]._lastActivations = 0.0;
        l->_train_array[k]. _ActiveinputIds = 0;
    }

    (load) ?  layer_load(l, path) : layer_randinit(l);               /* Set weights, bias, ADAM vel, ADAM mom */

    for (size_t i = 0; i < noOfNodes; i++) {                         /* Set node-specific pointers for convenience */
        size_t index = prevLayerNumOfNodes * i;
        node_update(&l->_Nodes[i], i, type, batchsize, l->_train_array,
            &l->_weights[index], &l->_adamAvgMom[index], &l->_adamAvgVel[index], &l->_adamT[index], 
            &l->_bias[i], &l->_adamAvgMombias[i], &l->_adamAvgVelbias[i]);
    }

    layer_rehash(l);
    return l;
}

void layer_delete(Layer *l) {
    lsht_delete(l->_hashTables);
    dwtahash_delete(l->_dwtaHasher);
    size_t buffer_size = layer_size(l->_noOfNodes, l->_prevLayerNumOfNodes, l->_batchsize);
    fprintf(stderr, "Freeing %ld bytes for Layer %d at %p\n", buffer_size, l->_layerID, l);
    myunmap(l, buffer_size);
}

void layer_rehash(Layer *l) {
    lsht_clear(l->_hashTables);
    for (size_t i = 0; i < l->_noOfNodes; i++) {
        size_t index = l->_prevLayerNumOfNodes * i;
        int *hashes = dwtahash_getHashEasy(l->_dwtaHasher, &l->_weights[index], l->_prevLayerNumOfNodes);
        lsht_add(l->_hashTables, hashes, i);
        free(hashes);
    }
}

/* 
 * Kaiming initialization preferable for faster covergence in deep networks 
 */
void layer_randinit(Layer *l) {
    float ksd = sqrt(1.0/l->_prevLayerNumOfNodes);
    for (size_t i = 0; i < l->_noOfNodes; i++) {
        size_t fano = i * l->_prevLayerNumOfNodes;
        for (size_t j = 0; j < l->_prevLayerNumOfNodes; j++) l->_weights[fano+j] = myrand_norm(0.0,ksd);
        l->_bias[i] = 0.0;
        l->_adamAvgMombias[i] = 0.0;
        l->_adamAvgVelbias[i] = 0.0;
    }
}

void layer_updateHasher(Layer *l) {
    dwtahash_delete(l->_dwtaHasher);
    l->_dwtaHasher = dwtahash_new(l->_K * l->_L, l->_prevLayerNumOfNodes);
}

void layer_updateRandomNodes(Layer *l) { 
    myrand_shuffle(l->_randNode, l->_noOfNodes); 
}

int layer_get_prediction(Layer *l, int *activeNodesOut, int lengthOut, int inputID) {
    assert(l->_type == Softmax);
    int predict_class = -1;
    float max_act = INT_MIN;
    for(int k = 0; k < lengthOut; k++) {
        float cur_act = node_get_last_activation(&l->_Nodes[activeNodesOut[k]], inputID);
        if (max_act < cur_act) {
            max_act = cur_act;
            predict_class = activeNodesOut[k];
        }
    }
    return predict_class;
}

/* Expects output arrays of size l->_noOfNodes, though lengthOut could be smaller */ 
/* Seems safe to call in parallel, once per inputID in a batch, provided: 
 *   - no other thread modifies label
 *   - activeNodesIn/Out, activeValuesIn/Out, and lengthOut are dedicated to this call
 *   - no other thread reads or modifies l->_normalizationConstants[inputID]
 *   - no other thread reads or modifies n->_train[inputID] for any node in the layer
 */
int layer_fwdprop(Layer *l, 
    int *activeNodesIn, float *activeValuesIn, int lengthIn,        /* from previous layer */
    int *activeNodesOut, float *activeValuesOut, int *lengthOut,    /* to next layer */
    int inputID, int *label, int labelsize, float Sparsity) {

    int retrievals = 0;
    int len;

    assert(Sparsity <= 1.0);
    if(Sparsity == 1.0) {
        len = l->_noOfNodes;
        *lengthOut = len;
        for (int i = 0; i < len; i++) activeNodesOut[i] = i;
    }
    else {
        Histo *counts = NULL;                              /* active candidates counts _MUST_ NULL init */
        Histo *cur = NULL, *tmp = NULL;                    /* for use with HASH_ITER */  

        if (l->_type == Softmax && labelsize > 0) {        /* ensure label node is in candidates */
            for (int i = 0; i < labelsize; i++) ht_put(&counts, label[i], l->_L);
        }

        int *hashes = dwtahash_getHash(l->_dwtaHasher, activeNodesIn, activeValuesIn, lengthIn);
        lsht_retrieve_histogram(l->_hashTables, hashes, &counts); /* get candidates from lsht */

        assert(((MINACTIVE > 0) && (THRESH == 0)) || ((THRESH > 0) && (MINACTIVE == 0)));
        if (THRESH > 1) {                                  /* drop candidates with counts < THRESH */
            HASH_ITER(hh, counts, cur, tmp) { 
                if (cur->value < THRESH) ht_del(&counts, &cur); 
            }
        }
        retrievals = ht_size(&counts);                     /* retrieved actives after any thresholding */

        if (MINACTIVE > 0) {                               /* add randomly to get min(l->_noOfNodes,MINACTIVE) */
            size_t st = myrand_unif() % l->_noOfNodes; 
            for (size_t i = st; i < l->_noOfNodes; i++) {  /* pick starting from a random index each time */
                if (ht_size(&counts) >= MINACTIVE) break;
                ht_put(&counts, i, 0);
            }
            for (size_t i = 0; i < st; i++) {              /* loop around to pick more if needed */
                if (ht_size(&counts) >= MINACTIVE) break;
                ht_put(&counts, i, 0);
            }
        }

        len = ht_size(&counts);                              /* actives after adding any randomly */
        *lengthOut = len;

        int i=0;
        HASH_ITER(hh, counts, cur, tmp) {
            activeNodesOut[i] = cur->key;
            i++;
        }
        assert(i == len);
        ht_destroy(&counts);
    }

    for (int i = 0; i < len; i++) {     /* get activation for all active nodes in layer */
        activeValuesOut[i] = node_get_activation(&l->_Nodes[activeNodesOut[i]], 
            activeNodesIn, activeValuesIn, lengthIn, inputID);
    }

    if(l->_type == Softmax) {
        float maxValue = 0;
        l->_normalizationConstants[inputID] = 0;
        for (int i = 0; i < len; i++) {
            if(activeValuesOut[i] > maxValue) maxValue = activeValuesOut[i];
        }
        for (int i = 0; i < len; i++) {
            float realActivation = exp(activeValuesOut[i] - maxValue);
            activeValuesOut[i] = realActivation;
            node_set_last_activation(&l->_Nodes[activeNodesOut[i]], inputID, realActivation);
            l->_normalizationConstants[inputID] += realActivation;
        }
    }

    return retrievals;
}

void layer_compute_softmax_stats(Layer *l, int *thisLayActiveIds, int thisLayActLen,
    float normalizationConstant, int inputID, int batchsize, int *label, int labelsize) {
    for(int i=0; i < thisLayActLen; i++) {
        Node *n = &(l->_Nodes[thisLayActiveIds[i]]);
        node_compute_softmax_stats(n, normalizationConstant, inputID, batchsize, label, labelsize);
    }
}

void layer_backprop(Layer *l, int *thisLayActiveIds, int thisLayActLen, Layer *prevLay,
    int *prevLayerActiveNodeIds, int prevLayerActiveNodeSize, float learningRate, int inputID) {
    Node *prevLayerNodeArray = prevLay->_Nodes;
    for(int i=0; i < thisLayActLen; i++) {
        Node *n = &(l->_Nodes[thisLayActiveIds[i]]);
        node_backprop(n, prevLayerNodeArray, prevLayerActiveNodeIds, prevLayerActiveNodeSize, 
            learningRate, inputID);
    }
}

void layer_backprop_firstlayer(Layer *l, int *thisLayActiveIds, int thisLayActLen,
    int *nnzindices, float *nnzvalues, int nnzSize, float learningRate, int inputID) {
    for(int i=0; i < thisLayActLen; i++) {
        Node *n = &(l->_Nodes[thisLayActiveIds[i]]);
        node_backprop_firstlayer(n, nnzindices, nnzvalues, nnzSize, learningRate, inputID);
    }
}

void layer_adam(Layer *l, float lr, int ratio) {
#pragma omp parallel for
    for (size_t m = 0; m < l->_noOfNodes; m++) 
        node_adam(&l->_Nodes[m], l->_prevLayerNumOfNodes, l->_batchsize, lr, ratio);
}

inline static void __attribute__((always_inline)) layer_rw(Layer *l, char *path, bool load) {
    char fn[1024];
    size_t len = strlen(path);
    assert(len < 1000);
    strcpy(fn, path);
    void (*rwfn)(float*,size_t,size_t,char*);
    rwfn = load ? myload_fnpy : mysave_fnpy;
    sprintf(fn+len, "/b_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_bias, l->_noOfNodes, 1, fn);
    sprintf(fn+len, "/amb_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_adamAvgMombias, l->_noOfNodes, 1, fn);
    sprintf(fn+len, "/avb_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_adamAvgVelbias, l->_noOfNodes, 1, fn);

    sprintf(fn+len, "/w_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_weights, l->_noOfNodes, l->_prevLayerNumOfNodes, fn);
    sprintf(fn+len, "/av_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_adamAvgVel, l->_noOfNodes, l->_prevLayerNumOfNodes, fn);
    sprintf(fn+len, "/am_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_adamAvgMom, l->_noOfNodes, l->_prevLayerNumOfNodes, fn);
    fprintf(stderr, "%s parameters for layer %d\n", load ? "Loaded" : "Saved", l->_layerID);
}

void layer_load(Layer *l, char *path) { layer_rw(l, path, true); }

void layer_save(Layer *l, char *path) { layer_rw(l, path, false); }

