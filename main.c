#include "hdefs.h"

static void test_mt() {
    int i,N = 10000;
    double totl;
    for (i=0, totl=0.0; i<N; i++) totl+=myrand_unif();
    printf("\nFor %d samples of myrand_unif() -- ", N);
    printf("Expected mean: %d  Observed mean: %d\n", INT_MAX/2, (int)(totl/N));
}

static void test_norm() {
    int i,N = 10000;
    float samp[10000];
    float totl, mean, stdv = 0;
    printf("\nFor %d samples of myrand_norm() -- ", N);
    for (i=0, totl=0.0; i<N; i++) {
        samp[i] = myrand_norm(0.0, 0.01);
        totl+=samp[i];
    }
    mean = totl / N;
    samp[i] = samp[i] - mean;
    for (i=0, totl=0.0; i<N; i++) totl += samp[i] * samp[i];
    stdv = sqrt(totl / N);
    printf("Expected mean,stdv: 0.0,0.01  Observed mean,stdv: %.4f,%.4f\n", mean, stdv);
}

static void test_myshuffle() {
    int i, x[100];
    for (i = 0; i < 100; i++) x[i] = i;
    myshuffle(x, 100);
    printf("\nOutput shuffle of [0:99]\n");
    for (i = 0; i < 100; i++) {
        printf("%d", x[i]);
        printf("%s", (i%20==19) ? "\n" : " ");
    }
}

static void ht_update(khash_t(hist) *h) {
    int isnew;
    khiter_t k;
    for(int i = 2; i<10; i++) {
       k = kh_put(hist, h, i, &isnew);
       kh_value(h, k) = (isnew) ? 1 : (kh_value(h, k) + 1);
    }
    for(int i = 4; i<8; i++) {
       k = kh_put(hist, h, i, &isnew);
       kh_value(h, k) = (isnew) ? 1 : (kh_value(h, k) + 1);
    }
}

static void test_hashtable() {
    khiter_t k;
    khash_t(hist) *h = kh_init(hist);
    ht_update(h);
    printf("num keys: %d\n", kh_size(h));
    for (k = kh_begin(h); k != kh_end(h); ++k)
       if (kh_exist(h, k)) printf("key=%d, value=%ld\n", kh_key(h,k), kh_value(h, k));
    kh_destroy(hist, h);
}

static void test_wrnpy() {
    float ifarr[20];
    float ofarr[20];

    printf("\nTest saving and loading of 1D and 2D float arrays as npy files\n");
    for (int i=0; i<20; i++) ifarr[i] = i;

    mysave_fnpy(ifarr, false, 20, 1, "/tmp/float_20.npy");
    for (int i=0; i<20; i++) ofarr[i] = 0;
    myload_fnpy(ofarr, false, 20, 1, "/tmp/float_20.npy");
    for (int i=0; i<20; i++) assert(ifarr[i] == ofarr[i]);

    mysave_fnpy(ifarr, true, 1, 20, "/tmp/float_1x20.npy");
    for (int i=0; i<20; i++) ofarr[i] = 0;
    myload_fnpy(ofarr, true, 1, 20, "/tmp/float_1x20.npy");
    for (int i=0; i<20; i++) assert(ifarr[i] == ofarr[i]);

    mysave_fnpy(ifarr, true, 20, 1, "/tmp/float_20x1.npy");
    for (int i=0; i<20; i++) ofarr[i] = 0;
    myload_fnpy(ofarr, true, 20, 1, "/tmp/float_20x1.npy");
    for (int i=0; i<20; i++) assert(ifarr[i] == ofarr[i]);

    mysave_fnpy(ifarr, true, 4, 5, "/tmp/float_4x5.npy");
    for (int i=0; i<20; i++) ofarr[i] = 0;
    myload_fnpy(ofarr, true, 4, 5, "/tmp/float_4x5.npy");
    for (int i=0; i<20; i++) assert(ifarr[i] == ofarr[i]);
}

static void test_dwtahash_lsht() {
    printf("\nTesting DWTAHash with LSHT\n");
    int *hashes;
    DWTAHash *d = dwtahash_new(6*50, 128);
    LSHT *l = lsht_new(6,50,18);

    float weights[10*128];
    for(int i = 0; i < 10*128; i++) weights[i] = myrand_norm(0,0.01);
    for(int j = 0; j < 10; j++) {
        hashes = dwtahash_getHashEasy(d, &weights[j*128], 128);
        lsht_add(l, hashes, j);
        free(hashes);
    }

    int ids[128];
    float myweights[128];
    for(int i = 128; i >= 0; i--) ids[i] = i;
    for(int i = 0; i < 128; i++) myweights[i] = weights[(4*128) + i];

    khiter_t k;
    khash_t(hist) *h = kh_init(hist);

    hashes = dwtahash_getHash(d, ids, myweights, 128);
    lsht_retrieve_histogram(l, hashes, h);

    printf("Expecting %d with count %d\n", 4, 50); 
    for (k = kh_begin(h); k != kh_end(h); ++k) {
        if (kh_exist(h, k)) {
            printf("index : %d, count: %ld\n", kh_key(h,k), kh_value(h, k));
        }
    }
    free(hashes);
    kh_destroy(hist, h);

    lsht_delete(l);
    dwtahash_delete(d);
}

static void test_layer(bool io) {
    printf("\nTesting Layer \n");
    Layer *l = layer_new(670091, 128, 1, ReLU, 1024, 6, 50, 18, false, NULL);
    if (io) {
       layer_save(l, "/tmp");
       layer_load(l, "/tmp");
    }
    layer_delete(l);
}

static void test_network(bool save, bool reload) {
    printf("\nBuilding network from scratch \n");
    Config *cfg = config_new("sampleconfig.json");
    Network *n = network_new(cfg, false);

    // training here

    if (save) {
       printf("Saving configuration and layer parameters\n");
       config_save(n->_cfg, "./data/config.json");
       network_save_params(n);
    }

    network_delete(n);
    config_delete(cfg);

    if (reload) {
        printf("Loading network from saved configuration and layer parameters\n");
        cfg = config_new("./data/config.json");
        n = network_new(cfg, true);
        network_delete(n);
        config_delete(cfg);
    }
}

int main(int argc, char *argv[]) {
/*
    test_mt();
    test_norm();
    test_myshuffle();
    test_hashtable();
    test_wrnpy();
    test_dwtahash_lsht();
    test_layer(true);
    test_network(true, true);
*/
    test_network(true, true);
    return 0;
}
