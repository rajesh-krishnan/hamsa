#include "hdefs.h"

Layer *layer_new(size_t noOfNodes, int prevLayerNumOfNodes, int layerID, NodeType type, int batchsize, 
    int K, int L, int RangePow, bool load, char *path) {
    size_t fano = noOfNodes * prevLayerNumOfNodes;
    Layer *l = (Layer *) mymap(sizeof(Layer));

    l->_noOfNodes = noOfNodes;
    l->_prevLayerNumOfNodes = prevLayerNumOfNodes;
    l->_layerID = layerID;
    l->_type = type;
    l->_batchsize = batchsize;
    l->_K = K;
    l->_L = L;
    l->_RangePow = RangePow;

    l->_Nodes       = (Node *)  mymap(noOfNodes * sizeof(Node));
    l->_train_array = (Train *) mymap(noOfNodes * batchsize * sizeof(Train));
    l->_weights     = (float *) mymap(fano * sizeof(float));
    l->_bias        = (float *) mymap(noOfNodes * sizeof(float));
    l->_adamAvgMom  = (float *) mymap(fano * sizeof(float));
    l->_adamAvgVel  = (float *) mymap(fano * sizeof(float));
    l->_adamT       = (float *) mymap(fano * sizeof(float));
    l->_randNode    = (int *)   mymap(noOfNodes * sizeof(int));
    l->_hashTables  = lsht_new(K, L, RangePow);
    l->_dwtaHasher  = dwtahash_new(K * L, prevLayerNumOfNodes);

    if (type == Softmax) l->_normalizationConstants = (float *) mymap(batchsize * sizeof(float));
    else                 l->_normalizationConstants = NULL;

    for (size_t n = 0; n < noOfNodes; n++) l->_randNode[n] = n;
    myshuffle(l->_randNode, noOfNodes);

    (load) ?  layer_load(l, path) : layer_randinit(l);

    for (size_t i = 0; i < noOfNodes; i++) {
        size_t index = prevLayerNumOfNodes * i;
        node_update(&l->_Nodes[i], i, type, batchsize,
            &l->_weights[index], &l->_bias[i], &l->_adamAvgMom[index], &l->_adamAvgVel[index], &l->_adamT[index], 
            l->_train_array);
    }

    layer_rehash(l);
    return l;
}

void layer_delete(Layer *l) {
    size_t fano = l->_noOfNodes * l->_prevLayerNumOfNodes;
    lsht_delete(l->_hashTables);
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

void layer_rehash(Layer *l) {
    lsht_clear(l->_hashTables);
    for (size_t i = 0; i < l->_noOfNodes; i++) {
        size_t index = l->_prevLayerNumOfNodes * i;
        layer_addToHashTable(l, &l->_weights[index], l->_prevLayerNumOfNodes, i);
    }
}

/* 
 * Kaiming initialization preferable for faster covergence in deep networks 
 */
void layer_randinit(Layer *l) {
    float ksd = sqrt(1.0/l->_prevLayerNumOfNodes);
#pragma omp parallel for
    for (size_t i = 0; i < l->_noOfNodes; i++) {
        size_t fano = i * l->_prevLayerNumOfNodes;
        for (size_t j = 0; j < l->_prevLayerNumOfNodes; j++) l->_weights[fano+j] = myrand_norm(0.0,ksd);
        l->_bias[i] = 0.0;
    }
}

inline static void __attribute__((always_inline)) layer_rw(Layer *l, char *path, bool load) {
    char fn[1024];
    size_t len = strlen(path);
    assert(len < 1000);
    strcpy(fn, path);
    void (*rwfn)(float*,bool,size_t,size_t,char*);
    rwfn = load ? myload_fnpy : mysave_fnpy;
    sprintf(fn+len, "/b_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_bias, false, l->_noOfNodes, 1, fn);
    sprintf(fn+len, "/w_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_weights, true, l->_noOfNodes, l->_prevLayerNumOfNodes, fn);
    /* could also save ADAM parameters here */
    fprintf(stderr, "%s parameters for layer %d\n", load ? "Loaded" : "Saved", l->_layerID);
}

void layer_load(Layer *l, char *path) { layer_rw(l, path, true); }

void layer_save(Layer *l, char *path) { layer_rw(l, path, false); }

void layer_updateHasher(Layer *l) {
    dwtahash_delete(l->_dwtaHasher);
    l->_dwtaHasher = dwtahash_new(l->_K * l->_L, l->_prevLayerNumOfNodes);
}

void layer_updateRandomNodes(Layer *l) { myshuffle(l->_randNode, l->_noOfNodes); }

void layer_addToHashTable(Layer *l, float* weights, int length, int id) {
    int *hashes = dwtahash_getHashEasy(l->_dwtaHasher, weights, length);
    lsht_add(l->_hashTables, hashes, id);
    free(hashes);
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
 *   - no other thread reads or modifies n->_train[inputID]._ActiveinputIds for any node in layer
 *   - no other thread reads or modifies n->_train[inputID]._lastActivations for any node in layer
 */
int layer_forwardPropagate(Layer *l, 
    int *activeNodesIn, float *activeValuesIn, int lengthIn,        /* from previous layer */
    int *activeNodesOut, float *activeValuesOut, int *lengthOut,    /* to next layer */
    int inputID, int *label, int labelsize, float Sparsity) {

    int retrievals = 0;
    int len;

    if(Sparsity == 1.0) {
        len = l->_noOfNodes;
        *lengthOut = len;
        for (int i = 0; i < len; i++) activeNodesOut[i] = i;
    }
    else {
        int isnew;
        khiter_t k;
        khash_t(hist) *h = kh_init(hist);                  /* to store active candidates with counts */

        if (l->_type == Softmax && labelsize > 0) {        /* ensure label node is in candidates */
            for (int i = 0; i < labelsize; i++) {
                k = kh_put(hist, h, i, &isnew);
                kh_value(h, k) = l->_L;
            }
        }

        int *hashes = dwtahash_getHash(l->_dwtaHasher, activeNodesIn, activeValuesIn, lengthIn);
        lsht_retrieve_histogram(l->_hashTables, hashes, h); /* get candidates from hashtable */

        assert(((MINACTIVE > 0) && (THRESH == 0)) || ((THRESH > 0) && (MINACTIVE == 0)));

        if (THRESH > 1) {                                  /* drop candidates with counts < THRESH */
            for (k = kh_begin(h); k != kh_end(h); ++k) 
                if (kh_exist(h, k) && (kh_value(h, k) < THRESH)) kh_del(hist, h, k);
        }
        retrievals = kh_size(h);                           /* retrieved actives after any thresholding */

        if (MINACTIVE > 0) {                               /* add randomly to get min(l->_noOfNodes,MINACTIVE) */
            size_t st = myrand_unif() % l->_noOfNodes; 
            for (size_t i = st; i < l->_noOfNodes; i++) {  /* pick starting from a random index each time */
                if (kh_size(h) >= MINACTIVE) break;
                k = kh_put(hist, h, i, &isnew);
                if (isnew) kh_value(h, k) = 0;
            }
            for (size_t i = 0; i < st; i++) {              /* loop around to pick more if needed */
                if (kh_size(h) >= MINACTIVE) break;
                k = kh_put(hist, h, i, &isnew);
                if (isnew) kh_value(h, k) = 0;
            }
        }

        len = kh_size(h);                                  /* actives after adding any randomly */
        *lengthOut = len;

        int i=0;
        for (k = kh_begin(h); k != kh_end(h); ++k) {
            if (kh_exist(h, k)) {
                activeNodesOut[i] = kh_key(h, k);
                i++;
            }
        }
        assert(i == len);
        kh_destroy(hist, h);
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

void layer_adam(Layer *l, float lr, int ratio) {
#pragma omp parallel for
    for (size_t m = 0; m < l->_noOfNodes; m++) node_adam(&l->_Nodes[m], l->_prevLayerNumOfNodes, lr, ratio);
}
