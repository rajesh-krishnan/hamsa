#pragma once
#define BUCKETSIZE 128
#define BINSIZE 8

#pragma pack(push,1)
typedef struct _struct_dwtahash {
    int _numhashes;
    int _rangePow;
    int _lognumhash;
    int _permute;
    int _randHash[2];
    int *_indices;
    int *_pos;
} DWTAHash;

typedef struct _struct_bucket {
    int count;
    int arr[BUCKETSIZE];
} Bucket;

typedef struct _struct_lsht {
    Bucket **_bucket;
    int _K;
    int _L;
    int _RangePow;
} LSHT;

typedef enum { ReLU = 1, Softmax = 2 } NodeType;

typedef struct _struct_train {
    float _lastDeltaforBPs;
    float _lastActivations;
    int _ActiveinputIds;
} Train;

typedef struct _struct_node {
    size_t _IDinLayer;
    NodeType _type;
    Train *_train;
    float *_weights;
    float *_adamAvgMom;
    float *_adamAvgVel;
    float *_t;
    float *_bias;
    float *_adamAvgMombias;
    float *_adamAvgVelbias;
} Node;

typedef struct _struct_layer {
    size_t _noOfNodes;
    int _prevLayerNumOfNodes;
    int _layerID;
    NodeType _type;
    int _batchsize;
    int _K;
    int _L;
    int _RangePow;
    Node *_Nodes;
    Train *_train_array;
    float *_weights;
    float *_adamAvgMom;
    float *_adamAvgVel;
    float *_adamT;
    float *_bias;
    float *_adamAvgMombias;
    float *_adamAvgVelbias;
    int *_randNode;
    float *_normalizationConstants;
    LSHT *_hashTables;
    DWTAHash *_dwtaHasher;
} Layer;

typedef struct _struct_config {
    int numLayer;
    int *sizesOfLayers;
    NodeType *layersTypes;
    int *RangePow;
    int *K;
    int *L;
    float *Sparsity;
    int Batchsize;
    int Rehash;
    int Rebuild;
    int Reperm;
    int InputDim;
    int totRecords;
    int totRecordsTest;
    float Lr;
    int Epoch;
    int Stepsize;
    char *trainData;
    char *testData;
    char *loadPath;
    char *savePath;
    char *logFile;
} Config;

typedef struct _struct_network {
    Layer **_hiddenlayers;
    Config *_cfg;
} Network;
#pragma pack(pop)

Config *config_new(const char *cfgFile);
void config_delete(Config *cfg);
void config_save(Config *cfg, const char *cfgFile);

Network *network_new(Config *cfg, bool loadParams);
void network_delete(Network *n);
void network_load_params(Network *n);
void network_save_params(Network *n);
int network_infer(Network *n, int **inIndices, float **inValues, int *inLength, int **blabels, int *blabelsize);
void network_train(Network *n, int **inIndices, float **inValues, int *inLength, int **blabels, int *blabelsize,
    int iter, bool reperm, bool rehash, bool rebuild);

Layer *layer_new(size_t noOfNodes, int prevLayerNumOfNodes, int layerID, NodeType type, int batchsize, 
    int K, int L, int RangePow, bool load, char *path);
void layer_delete(Layer *l);
void layer_rehash(Layer *l);
void layer_randinit(Layer *l);
void layer_updateHasher(Layer *l);
void layer_updateRandomNodes(Layer *l);
int layer_get_prediction(Layer *l, int *activeNodesOut, int lengthOut, int inputID);
int layer_fwdprop(Layer *l, 
    int *activeNodesIn, float *activeValuesIn, int lengthIn, 
    int *activeNodesOut, float *activeValuesOut, int *lengthOut, 
    int inputID, int *label, int labelsize, float Sparsity);
void layer_compute_softmax_stats(Layer *l, int *thisLayActiveIds, int thisLayActLen,
    float normalizationConstant, int inputID, int batchsize, int *label, int labelsize);
void layer_backprop(Layer *l, int *thisLayActiveIds, int thisLayActLen, Layer *prevLay,
    int *prevLayerActiveNodeIds, int prevLayerActiveNodeSize, float learningRate, int inputID);
void layer_backprop_firstlayer(Layer *l, int *thisLayActiveIds, int thisLayActLen,
    int *nnzindices, float *nnzvalues, int nnzSize, float learningRate, int inputID);
void layer_adam(Layer *l, float lr, int ratio);
void layer_load(Layer *l, char *path);
void layer_save(Layer *l, char *path);
