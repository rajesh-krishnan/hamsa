#pragma once
#include "myhelper.h"
#include "node.h"

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

Config *config_new(char *cfgFile);
void config_delete(Config *cfg);
void config_save(Config *cfg, char *cfgFile);

