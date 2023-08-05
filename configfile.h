#pragma once
#include "myhelper.h"

typedef struct _struct_config {
    int numLayer;
    int *sizesOfLayers;
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

Config *config_new();
void config_delete(Config *cfg);
void string_to_config(char *jstr, Config *cfg);
void config_to_string(Config *cfg, char *ostr, int maxlen);
