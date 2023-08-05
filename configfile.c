#include "configfile.h"

Config *config_new() {
    Config *cfg = (Config *) malloc(sizeof(Config));
    cfg->sizesOfLayers = NULL;
    cfg->RangePow = NULL;
    cfg->K = NULL;
    cfg->L = NULL;
    cfg->Sparsity = NULL;
    cfg->trainData = NULL;
    cfg->testData = NULL;
    cfg->loadPath = NULL;
    cfg->savePath = NULL;
    cfg->logFile = NULL;
    return cfg;
}

void config_delete(Config *cfg) {
#define FREE_IF_NOT_NULL(X) if (X) free(X)
    FREE_IF_NOT_NULL(cfg->sizesOfLayers);
    FREE_IF_NOT_NULL(cfg->RangePow);
    FREE_IF_NOT_NULL(cfg->K);
    FREE_IF_NOT_NULL(cfg->L);
    FREE_IF_NOT_NULL(cfg->Sparsity);
    FREE_IF_NOT_NULL(cfg->trainData);
    FREE_IF_NOT_NULL(cfg->testData);
    FREE_IF_NOT_NULL(cfg->loadPath);
    FREE_IF_NOT_NULL(cfg->savePath);
    FREE_IF_NOT_NULL(cfg->logFile);
    free(cfg);
#undef FREE_IF_NOT_NULL
}

void config_to_string(Config *cfg, char *jstr, int maxlen) {
    int tmp, curlen = 0;

#define ADD_ITEM(X) tmp=strlen((X));assert((curlen+tmp)<maxlen);memcpy(jstr+curlen,(X),tmp);curlen+=tmp;

    ADD_ITEM("{\n");
    ADD_ITEM("  \"numLayer\":        ");

    ADD_ITEM(",\n");
    ADD_ITEM("  \"sizesOfLayers\":   [");

    ADD_ITEM("],\n");
    ADD_ITEM("  \"RangePow\":        [");

    ADD_ITEM("],\n");
    ADD_ITEM("  \"K\":               [");

    ADD_ITEM("],\n");
    ADD_ITEM("  \"L\":               [");

    ADD_ITEM("],\n");
    ADD_ITEM("  \"Sparsity\":        [");

    ADD_ITEM("],\n");
    ADD_ITEM("  \"Batchsize\":       ");

    ADD_ITEM(",\n");
    ADD_ITEM("  \"Rehash\":          ");

    ADD_ITEM(",\n");
    ADD_ITEM("  \"Rebuild\":         ");

    ADD_ITEM(",\n");
    ADD_ITEM("  \"InputDim\":        ");

    ADD_ITEM(",\n");
    ADD_ITEM("  \"totRecords\":      ");

    ADD_ITEM(",\n");
    ADD_ITEM("  \"totRecordsTest\":  ");

    ADD_ITEM(",\n");
    ADD_ITEM("  \"Lr\":              ");

    ADD_ITEM(",\n");
    ADD_ITEM("  \"Epoch\":           ");

    ADD_ITEM(",\n");
    ADD_ITEM("  \"Stepsize\":        ");

    ADD_ITEM(",\n");
    ADD_ITEM("  \"trainData\":       \"");

    ADD_ITEM("\",\n");
    ADD_ITEM("  \"testData\":        \"");

    ADD_ITEM("\",\n");
    ADD_ITEM("  \"loadpath\":        \"");

    ADD_ITEM("\",\n");
    ADD_ITEM("  \"savePath\":        \"");

    ADD_ITEM("\",\n");
    ADD_ITEM("  \"logFile\":         \"");

    ADD_ITEM("\"\n}\n");
    ADD_ITEM("\0");
#undef ADD_ITEM
}

void string_to_config(char *jstr, Config *cfg) {
    size_t len = strlen(jstr) + 1;
    char *tmp = (char *) malloc(len * sizeof(char));

    memcpy(tmp, jstr, len);
    json_t mem[100000];
    json_t const* json = json_create( tmp, mem, sizeof mem / sizeof *mem );
    if ( !json ) {
        puts("Error json create.");
        exit(EXIT_FAILURE);
    }

    // extract from json and put in config
    // sanity check values
    free(tmp);
}

