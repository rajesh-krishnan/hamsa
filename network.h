#pragma once
#include "layer.h"
#include "configfile.h"

typedef struct _struct_network {
    Layer **_hiddenlayers;
    Config *_cfg;
} Network;

Network *network_new(Config *cfg, bool loadParams);
void network_delete(Network *n);
void network_load_params(Network *n);
void network_save_params(Network *n);
int network_infer(Network *n, int **inputIndices, float **inputValues, int *length, int **labels, int *labelsize);
void network_train(Network *n, int **inputIndices, float **inputValues, int *lengths, int **label, int *labelsize, 
    int iter, bool rehash, bool rebuild);

