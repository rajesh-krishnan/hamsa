#include "hdefs.h"
#include "tiny-json/tiny-json.h"

static char *dupstr(const char* s) {
    size_t slen = strlen(s);
    char *result = malloc(slen + 1);
    assert(result != NULL);
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

static void config_to_string(Config *cfg, char *ostr, int maxlen) {
#define ADD_ITEM(X) tmp=strlen((X));assert((curlen+tmp)<maxlen);memcpy(ostr+curlen,(X),tmp);curlen+=tmp;
    int tmp, curlen = 0;
    char buffer[10000];
    ADD_ITEM("{\n");
    ADD_ITEM("  \"numLayer\":        ");
    sprintf(buffer, "%d%s", cfg->numLayer, ",\n");
    ADD_ITEM(buffer);
    ADD_ITEM("  \"sizesOfLayers\":   [");
    for (int i=0; i < cfg->numLayer; i++) {
        sprintf(buffer, "%d%s", cfg->sizesOfLayers[i], (i==cfg->numLayer-1)?"],\n":",");
        ADD_ITEM(buffer);
    }
    ADD_ITEM("  \"layersTypes\":     [");
    for (int i=0; i < cfg->numLayer; i++) {
        sprintf(buffer, "%d%s", cfg->layersTypes[i], (i==cfg->numLayer-1)?"],\n":",");
        ADD_ITEM(buffer);
    }
    ADD_ITEM("  \"RangePow\":        [");
    for (int i=0; i < cfg->numLayer; i++) {
        sprintf(buffer, "%d%s", cfg->RangePow[i], (i==cfg->numLayer-1)?"],\n":",");
        ADD_ITEM(buffer);
    }
    ADD_ITEM("  \"K\":               [");
    for (int i=0; i < cfg->numLayer; i++) {
        sprintf(buffer, "%d%s", cfg->K[i], (i==cfg->numLayer-1)?"],\n":",");
        ADD_ITEM(buffer);
    }
    ADD_ITEM("  \"L\":               [");
    for (int i=0; i < cfg->numLayer; i++) {
        sprintf(buffer, "%d%s", cfg->L[i], (i==cfg->numLayer-1)?"],\n":",");
        ADD_ITEM(buffer);
    }
    ADD_ITEM("  \"Sparsity\":        [");
    for (int i=0; i < 2*cfg->numLayer; i++) {
        sprintf(buffer, "%g%s", cfg->Sparsity[i], (i==2*cfg->numLayer-1)?"],\n":",");
        ADD_ITEM(buffer);
    }
    ADD_ITEM("  \"Batchsize\":       ");
    sprintf(buffer, "%d%s", cfg->Batchsize, ",\n");
    ADD_ITEM(buffer);
    ADD_ITEM("  \"Rehash\":          ");
    sprintf(buffer, "%d%s", cfg->Rehash, ",\n");
    ADD_ITEM(buffer);
    ADD_ITEM("  \"Rebuild\":         ");
    sprintf(buffer, "%d%s", cfg->Rebuild, ",\n");
    ADD_ITEM(buffer);
    ADD_ITEM("  \"InputDim\":        ");
    sprintf(buffer, "%d%s", cfg->InputDim, ",\n");
    ADD_ITEM(buffer);
    ADD_ITEM("  \"totRecords\":      ");
    sprintf(buffer, "%d%s", cfg->totRecords, ",\n");
    ADD_ITEM(buffer);
    ADD_ITEM("  \"totRecordsTest\":  ");
    sprintf(buffer, "%d%s", cfg->totRecordsTest, ",\n");
    ADD_ITEM(buffer);
    ADD_ITEM("  \"Lr\":              ");
    sprintf(buffer, "%g%s", cfg->Lr, ",\n");
    ADD_ITEM(buffer);
    ADD_ITEM("  \"Epoch\":           ");
    sprintf(buffer, "%d%s", cfg->Epoch, ",\n");
    ADD_ITEM(buffer);
    ADD_ITEM("  \"Stepsize\":        ");
    sprintf(buffer, "%d%s", cfg->Stepsize, ",\n");
    ADD_ITEM(buffer);
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
    ostr[curlen]='\0';
#undef ADD_ITEM
}

static void string_to_config(char *jstr, Config *cfg) {
    int retr;
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
    x = json_getProperty(json, "sizesOfLayers");
    assert((x != NULL) && (json_getType(x) == JSON_ARRAY));
    cfg->sizesOfLayers = dupints(x, &retr);
    assert(retr == cfg->numLayer);
    x = json_getProperty(json, "layersTypes");
    assert((x != NULL) && (json_getType(x) == JSON_ARRAY));
    cfg->layersTypes = (NodeType *) dupints(x, &retr);
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
    free(tmp);
}

Config *config_new(char *cfgFile) {
    Config *cfg = (Config *) malloc(sizeof(Config));
    char *buffer;
    size_t length;
    FILE *f = fopen(cfgFile, "rb");
    assert(f != NULL);
    fseek(f, 0, SEEK_END);
    length = ftell(f);
    fseek (f, 0, SEEK_SET);
    buffer = malloc(length+1);
    if (buffer) {
        int x = fread(buffer, 1, length, f);
        assert(x == length);
        buffer[length] = '\0';
    }
    fclose(f);
    string_to_config(buffer, cfg);
    free(buffer);
    return cfg;
}

void config_save(Config *cfg, char *cfgFile) {
    char buffer[10000];
    config_to_string(cfg, buffer, 10000);
    FILE *f = fopen(cfgFile, "w");
    size_t length = strlen(buffer);
    if (f) {
        int x = fwrite(buffer, 1, length, f);
        assert(x == length);
    }
    fclose(f);
}

void config_delete(Config *cfg) {
#define FREE_IF_NOT_NULL(X) if (X) free(X)
    FREE_IF_NOT_NULL(cfg->sizesOfLayers);
    FREE_IF_NOT_NULL(cfg->layersTypes);
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

