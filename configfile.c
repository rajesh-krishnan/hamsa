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
    char buffer[10000];

#define ADD_ITEM(X) tmp=strlen((X));assert((curlen+tmp)<maxlen);memcpy(jstr+curlen,(X),tmp);curlen+=tmp;

    ADD_ITEM("{\n");
    ADD_ITEM("  \"numLayer\":        ");
    sprintf(buffer, "%d", cfg->numLayer);
    ADD_ITEM(buffer);

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
    sprintf(buffer, "%d", cfg->Batchsize);
    ADD_ITEM(buffer);

    ADD_ITEM(",\n");
    ADD_ITEM("  \"Rehash\":          ");
    sprintf(buffer, "%d", cfg->Rehash);
    ADD_ITEM(buffer);

    ADD_ITEM(",\n");
    ADD_ITEM("  \"Rebuild\":         ");
    sprintf(buffer, "%d", cfg->Rebuild);
    ADD_ITEM(buffer);

    ADD_ITEM(",\n");
    ADD_ITEM("  \"InputDim\":        ");
    sprintf(buffer, "%d", cfg->InputDim);
    ADD_ITEM(buffer);

    ADD_ITEM(",\n");
    ADD_ITEM("  \"totRecords\":      ");
    sprintf(buffer, "%d", cfg->totRecords);
    ADD_ITEM(buffer);

    ADD_ITEM(",\n");
    ADD_ITEM("  \"totRecordsTest\":  ");
    sprintf(buffer, "%d", cfg->totRecordsTest);
    ADD_ITEM(buffer);

    ADD_ITEM(",\n");
    ADD_ITEM("  \"Lr\":              ");
    sprintf(buffer, "%g", cfg->Lr);
    ADD_ITEM(buffer);

    ADD_ITEM(",\n");
    ADD_ITEM("  \"Epoch\":           ");
    sprintf(buffer, "%d", cfg->Epoch);
    ADD_ITEM(buffer);

    ADD_ITEM(",\n");
    ADD_ITEM("  \"Stepsize\":        ");
    sprintf(buffer, "%d", cfg->Stepsize);
    ADD_ITEM(buffer);

    ADD_ITEM(",\n");
    ADD_ITEM("  \"trainData\":       \"");
    ADD_ITEM(cfg->trainData);


    ADD_ITEM("\",\n");
    ADD_ITEM("  \"testData\":        \"");
    ADD_ITEM(cfg->testData);

    ADD_ITEM("\",\n");
    ADD_ITEM("  \"loadPath\":        \"");
    ADD_ITEM(cfg->loadPath);

    ADD_ITEM("\",\n");
    ADD_ITEM("  \"savePath\":        \"");
    ADD_ITEM(cfg->savePath);

    ADD_ITEM("\",\n");
    ADD_ITEM("  \"logFile\":         \"");
    ADD_ITEM(cfg->logFile);

    ADD_ITEM("\"\n}\n");
    ADD_ITEM("\0");
#undef ADD_ITEM
}

static char *dupstr(const char* s) {
  size_t slen = strlen(s);
  char *result = malloc(slen + 1);
  if(result == NULL) return NULL;
  memcpy(result, s, slen+1);
  return result;
}

static float *dupfloats(json_t const *x, int *count) {
    float buf[10000];
    int c = 0;
    for(json_t const *child = json_getChild(x); child != 0; child = json_getSibling(child)) {
        if (json_getType(child) == JSON_REAL) {
            buf[c] = (float) json_getReal(child);
        } 
        else if (json_getType(child) == JSON_INTEGER) {
            buf[c] = (float) json_getInteger(child);
        }
        else assert(1 == 0); /* force error */
        c++;
        assert (c < 10000);
    }
    *count = c;

    float *tmp = malloc(c * sizeof(float));
    assert(tmp != NULL);
    memcpy(tmp, buf, c*sizeof(float));
    return tmp;
}

static int *dupints(json_t const *x, int *count) {
    int buf[10000];
    int c = 0;
    for(json_t const *child = json_getChild(x); child != 0; child = json_getSibling(child)) {
        assert (json_getType(child) == JSON_INTEGER);
        buf[c] = (int) json_getInteger(child);
        c++;
    }
    *count = c;

    int *tmp = malloc(c * sizeof(int));
    assert(tmp != NULL);
    memcpy(tmp, buf, c*sizeof(int));
    return tmp;
}

void string_to_config(char *jstr, Config *cfg) {
    size_t len = strlen(jstr) + 1;
    char *tmp = (char *) malloc(len * sizeof(char));

    memcpy(tmp, jstr, len);
    json_t mem[100000];
    json_t const *json = json_create( tmp, mem, sizeof mem / sizeof *mem );
    if ( !json ) {
        puts("Error json create.");
        exit(EXIT_FAILURE);
    }

    assert(json_getType(json) == JSON_OBJ);
    json_t const *x;

    x = json_getProperty(json, "numLayer");
    assert((x != NULL) && (json_getType(x) == JSON_INTEGER));
    cfg->numLayer = (int) json_getInteger(x);

    int retr;
    x = json_getProperty(json, "sizesOfLayers");
    assert((x != NULL) && (json_getType(x) == JSON_ARRAY));
    cfg->sizesOfLayers = dupints(x, &retr);
    assert(retr == cfg->numLayer);

    x = json_getProperty(json, "RangePow");
    assert((x != NULL) && (json_getType(x) == JSON_ARRAY));
    cfg->RangePow = dupints(x, &retr);
    assert(retr == cfg->numLayer);

    x = json_getProperty(json, "K");
    assert((x != NULL) && (json_getType(x) == JSON_ARRAY));
    cfg->K = dupints(x, &retr);
    assert(retr == cfg->numLayer);

    x = json_getProperty(json, "L");
    assert((x != NULL) && (json_getType(x) == JSON_ARRAY));
    cfg->L = dupints(x, &retr);
    assert(retr == cfg->numLayer);

    x = json_getProperty(json, "Sparsity");
    assert((x != NULL) && (json_getType(x) == JSON_ARRAY));
    cfg->Sparsity = dupfloats(x, &retr);
    assert(retr == 2*cfg->numLayer);

    x = json_getProperty(json, "Batchsize");
    assert((x != NULL) && (json_getType(x) == JSON_INTEGER));
    cfg->Batchsize = (int) json_getInteger(x);

    x = json_getProperty(json, "Rehash");
    assert((x != NULL) && (json_getType(x) == JSON_INTEGER));
    cfg->Rehash = (int) json_getInteger(x);

    x = json_getProperty(json, "Rebuild");
    assert((x != NULL) && (json_getType(x) == JSON_INTEGER));
    cfg->Rebuild = (int) json_getInteger(x);

    x = json_getProperty(json, "InputDim");
    assert((x != NULL) && (json_getType(x) == JSON_INTEGER));
    cfg->InputDim = (int) json_getInteger(x);

    x = json_getProperty(json, "totRecords");
    assert((x != NULL) && (json_getType(x) == JSON_INTEGER));
    cfg->totRecords = (int) json_getInteger(x);

    x = json_getProperty(json, "totRecordsTest");
    assert((x != NULL) && (json_getType(x) == JSON_INTEGER));
    cfg->totRecordsTest = (int) json_getInteger(x);

    x = json_getProperty(json, "Lr");
    assert((x != NULL) && (json_getType(x) == JSON_REAL));
    cfg->Lr = (float) json_getReal(x);

    x = json_getProperty(json, "Epoch");
    assert((x != NULL) && (json_getType(x) == JSON_INTEGER));
    cfg->Epoch = (int) json_getInteger(x);

    x = json_getProperty(json, "Stepsize");
    assert((x != NULL) && (json_getType(x) == JSON_INTEGER));
    cfg->Stepsize = (int) json_getInteger(x);

    x = json_getProperty(json, "trainData");
    assert((x != NULL) && (json_getType(x) == JSON_TEXT));
    cfg->trainData = dupstr(json_getValue(x));

    x = json_getProperty(json, "testData");
    assert((x != NULL) && (json_getType(x) == JSON_TEXT));
    cfg->testData = dupstr(json_getValue(x));

    x = json_getProperty(json, "loadPath");
    assert((x != NULL) && (json_getType(x) == JSON_TEXT));
    cfg->loadPath = dupstr(json_getValue(x));

    x = json_getProperty(json, "savePath");
    assert((x != NULL) && (json_getType(x) == JSON_TEXT));
    cfg->savePath = dupstr(json_getValue(x));

    x = json_getProperty(json, "logFile");
    assert((x != NULL) && (json_getType(x) == JSON_TEXT));
    cfg->logFile = dupstr(json_getValue(x));


    // sanity check values

    free(tmp);
}

