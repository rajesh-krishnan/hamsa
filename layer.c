#include "layer.h"

Layer *layer_new(size_t noOfNodes, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, 
    int K, int L, int RangePow, bool load, char *path) {
    Layer *l = mymap(sizeof(Layer));

    l->_noOfNodes = noOfNodes;
    l->_previousLayerNumOfNodes = previousLayerNumOfNodes;
    l->_layerID = layerID;
    l->_type = type;
    l->_batchsize = batchsize;
    l->_K = K;
    l->_L = L;
    l->_RangePow = RangePow;

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
    for (size_t i = 0; i < noOfNodes; i++) {
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

/* 
 * Kaiming initialization preferable for faster covergence in deep networks 
 */
void layer_randinit(Layer *l) {
    float ksd = sqrt(1.0/l->_previousLayerNumOfNodes);
#pragma omp parallel for
    for (size_t i = 0; i < l->_noOfNodes; i++) {
        size_t fano = i * l->_previousLayerNumOfNodes;
        for (size_t j = 0; j < l->_previousLayerNumOfNodes; j++) l->_weights[fano+j] = myrand_norm(0.0,ksd);
        l->_bias[i] = 0.0;
    }
}

static void layer_rw(Layer *l, char *path, bool load) {
    char fn[1024];
    size_t len = strlen(path);
    assert(len < 1000);
    strcpy(fn, path);
    void (*rwfn)(float*,bool,size_t,size_t,char*);
    rwfn = load ? myload_fnpy : mysave_fnpy;
    sprintf(fn+len, "/b_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_bias, false, l->_noOfNodes, 1, fn);
    sprintf(fn+len, "/w_layer_%d.npy", l->_layerID);

    (*rwfn)(l->_weights, true, l->_noOfNodes, l->_previousLayerNumOfNodes, fn);
    sprintf(fn+len, "/am_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_adamAvgMom, true, l->_noOfNodes, l->_previousLayerNumOfNodes, fn);
    sprintf(fn+len, "/av_layer_%d.npy", l->_layerID);
    (*rwfn)(l->_adamAvgVel, true, l->_noOfNodes, l->_previousLayerNumOfNodes, fn);
    fprintf(stderr, "%s parameters for layer %d\n", load ? "Loaded" : "Saved", l->_layerID);
}

inline void __attribute__((always_inline)) layer_load(Layer *l, char *path) { layer_rw(l, path, true); }

inline void __attribute__((always_inline)) layer_save(Layer *l, char *path) { layer_rw(l, path, false); }

void layer_updateTable(Layer *l) {
    dwtahash_delete(l->_dwtaHasher);
    l->_dwtaHasher = dwtahash_new(l->_K * l->_L, l->_previousLayerNumOfNodes);
}

void layer_updateRandomNodes(Layer *l) { myshuffle(l->_randNode, l->_noOfNodes); }

void layer_addToHashTable(Layer *l, float* weights, int length, int id) {
    int *hashes = dwtahash_getHashEasy(l->_dwtaHasher, weights, length);
    lsh_add(l->_hashTables, hashes, id + 1);
    free(hashes);
}

/* ought to allocate memory outside */
/* ought to keep layer index confusion outside this function */

/* we deal with: # nodes of this layer, active values  */

/* computes activeValuesperLayer for the next layer */
/* compute activation based for RelU or Softmax */
/* ought to move this to node? */

int layer_forwardPropagate(Layer *l, int **activeNodes, float **activeValues, int *lengths, 
    int layerIndex, int inputID, int *label, int labelsize, float Sparsity, int iter) {

    int len;
    int retrievals = 0;

    if(Sparsity == 1.0) {
        len = l->_noOfNodes;
        lengths[layerIndex + 1] = len;
#if 0
        activeNodes[layerIndex + 1] = new int[len]; // XXX :assuming not intitialized;
#endif
        for (int i = 0; i < len; i++) activeNodes[layerIndex + 1][i] = i;
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

        int *hashes = dwtahash_getHash(l->_dwtaHasher, activeNodes[layerIndex], 
            activeValues[layerIndex], lengths[layerIndex]);

        lsh_retrieve_histogram(l->_hashTables, hashes, h); /* get candidates from hashtable */

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

        lengths[layerIndex + 1] = len;
#if 0
        activeNodes[layerIndex + 1] = new int[len]; // XXX: ought to alloc outside
#endif

        int i=0;
        for (k = kh_begin(h); k != kh_end(h); ++k) {
            if (kh_exist(h, k)) {
                activeNodes[layerIndex + 1][i] = k;
                i++;
            }
        }
        kh_destroy(hist, h);
    }

#if 0
    activeValues[layerIndex + 1] = new float[len]; // XXX: assuming its not initialized else memory leak;
#endif

    // find activation for all ACTIVE nodes in layer
    for (int i = 0; i < len; i++) {
        activeValues[layerIndex + 1][i] = 
            node_get_activation(&l->_Nodes[activeNodes[layerIndex + 1][i]], 
                activeNodes[layerIndex], activeValues[layerIndex], 
                lengths[layerIndex], inputID);
    }

    if(l->_type == Softmax) {
        float maxValue = 0;
        l->_normalizationConstants[inputID] = 0;
        for (int i = 0; i < len; i++) {
            if(activeValues[layerIndex + 1][i] > maxValue) {
                maxValue = activeValues[layerIndex + 1][i];
            }
        }
        for (int i = 0; i < len; i++) {
            float realActivation = exp(activeValues[layerIndex + 1][i] - maxValue);
            activeValues[layerIndex + 1][i] = realActivation;
            node_set_last_activation(&l->_Nodes[activeNodes[layerIndex + 1][i]], inputID, realActivation);
            l->_normalizationConstants[inputID] += realActivation;
        }
    }

    return retrievals;
}

