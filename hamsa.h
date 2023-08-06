#pragma once
#define BUCKETSIZE 128
#define BINSIZE 8

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

typedef struct _struct_lsh {
    Bucket ** _bucket;
    int _K;
    int _L;
    int _RangePow;
} LSH;

typedef enum { ReLU = 1, Softmax = 2 } NodeType;

typedef struct _struct_train {
    float _lastDeltaforBPs;
    float _lastActivations;
    float _lastGradients;
    int _ActiveinputIds;
} Train;

typedef struct _struct_node {
    size_t _IDinLayer;
    NodeType _type;
    int _currentBatchsize;
    float *_weights;
    float _bias;
    float *_adamAvgMom;
    float *_adamAvgVel;
    float* _t;
    Train *_train;
    int _activeInputs;
    float _tbias;
    float _adamAvgMombias;
    float _adamAvgVelbias;
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
    LSH *_hashTables;
    DWTAHash *_dwtaHasher;
    Train *_train_array;
    int *_randNode;
    float *_weights;
    float *_bias;
    float *_adamAvgMom;
    float *_adamAvgVel;
    float *_adamT;
    float *_normalizationConstants;
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

Config *config_new(char *cfgFile);
void config_delete(Config *cfg);
void config_save(Config *cfg, char *cfgFile);

Network *network_new(Config *cfg, bool loadParams);
void network_delete(Network *n);
void network_load_params(Network *n);
void network_save_params(Network *n);
int network_infer(Network *n, int **inputIndices, float **inputValues, int *length, int **labels, int *labelsize);
void network_train(Network *n, int **inputIndices, float **inputValues, int *lengths, int **label, int *labelsize, 
    int iter, bool rehash, bool rebuild);
