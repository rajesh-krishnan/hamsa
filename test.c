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
    stdv = sqrtf(totl / N);
    printf("Expected mean,stdv: 0.0,0.01  Observed mean,stdv: %.4f,%.4f\n", mean, stdv);
}

static void test_shuffle() {
    int i, x[100];
    for (i = 0; i < 100; i++) x[i] = i;
    myrand_shuffle(x, 100);
    printf("\nOutput shuffle of [0:99]\n");
    for (i = 0; i < 100; i++) {
        printf("%d", x[i]);
        printf("%s", (i%20==19) ? "\n" : " ");
    }
}

void test_uthash() {
    Histo *counts = NULL;

    printf("Size at init: %d\n", ht_size(&counts));
    for(int i = 2; i<10; i++) ht_incr(&counts, i);
    for(int i = 4; i<8; i++) ht_incr(&counts, i);
    ht_put(&counts, 29, 42);
    printf("Size after adding/incrementing items: %d\n", ht_size(&counts));
    ht_delkey(&counts, 5);
    printf("Size after deleting key: %d\n", ht_size(&counts));

#define THRESH1 2
    Histo *cur, *tmp;
    HASH_ITER(hh, counts, cur, tmp) {
        if (cur->value < THRESH1) ht_del(&counts, &cur);
    }
    printf("Size after deleting if value < %d: %d\n", THRESH1, ht_size(&counts));

    int x;
    HASH_ITER(hh, counts, cur, tmp) {
        x = cur->key;
        printf("Key: %d, Count: %ld\n", x, cur->value);
    }

    ht_destroy(&counts);
    printf("Size after destroying: %d\n", ht_size(&counts));
}

static void test_wrnpy() {
    float ifarr[20];
    float ofarr[20];

    printf("\nTest saving and loading of 1D and 2D float arrays as npy files\n");
    for (int i=0; i<20; i++) ifarr[i] = i;

    mysave_fnpy(ifarr, 1, 20, "/tmp/float_1x20.npy");
    for (int i=0; i<20; i++) ofarr[i] = 0;
    myload_fnpy(ofarr, 1, 20, "/tmp/float_1x20.npy");
    for (int i=0; i<20; i++) assert(ifarr[i] == ofarr[i]);

    mysave_fnpy(ifarr, 20, 1, "/tmp/float_20x1.npy");
    for (int i=0; i<20; i++) ofarr[i] = 0;
    myload_fnpy(ofarr, 20, 1, "/tmp/float_20x1.npy");
    for (int i=0; i<20; i++) assert(ifarr[i] == ofarr[i]);

    mysave_fnpy(ifarr, 4, 5, "/tmp/float_4x5.npy");
    for (int i=0; i<20; i++) ofarr[i] = 0;
    myload_fnpy(ofarr, 4, 5, "/tmp/float_4x5.npy");
    for (int i=0; i<20; i++) assert(ifarr[i] == ofarr[i]);
}

static void test_dwtahash_lsht() {
    printf("\nTesting DWTAHash with LSHT\n");
    int *hashes;
    DWTAHash *d = dwtahash_new(6*50, 128);
    LSHT *l = lsht_new(6,50,18);

    float weights[128];
    for(int i = 0; i < 128; i++) weights[i] = myrand_norm(0.0,0.01);
    hashes = dwtahash_getHashEasy(d, weights, 128);
    lsht_add(l, hashes, 3);
    free(hashes);

    int ids[128];
    for (int k=0; k<128; k++) ids[k] = k;

    Histo *counts = NULL; 
    Histo *cur, *tmp;

    hashes = dwtahash_getHash(d, ids, weights, 128);
    lsht_retrieve_histogram(l, hashes, &counts);
    printf("Query what was added, expecting %d with count %d\n", 3, 50); 
    HASH_ITER(hh, counts, cur, tmp) {
        printf("index : %d, count: %ld\n", cur->key, cur->value);
    }
    free(hashes);
    ht_destroy(&counts); counts = NULL;

    for(int i = 0; i < 128; i++) weights[i] = myrand_norm(0.0,0.01);
    hashes = dwtahash_getHash(d, ids, weights, 128);
    lsht_retrieve_histogram(l, hashes, &counts);
    printf("Random query, expecting no retrievals (or a few with low count)\n");
    HASH_ITER(hh, counts, cur, tmp) {
        printf("index : %d, count: %ld\n", cur->key, cur->value);
    }
    printf("Done\n");
    free(hashes);
    ht_destroy(&counts); counts = NULL;

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
    time_t t1, t2;

    t1 = time(NULL);
    printf("\nAllocating and initializing network ...\n"); 
    Config *cfg = config_new("sampleconfig.json");
    Network *n = network_new(cfg, false);
    t2 = time(NULL);
    printf("Allocating and initializing took %ld seconds\n", t2 - t1); 

    // training here

    if (save) {
        t1 = time(NULL);
        printf("Saving configuration and layer parameters ...\n");
        config_save(n->_cfg, "./data/config.json");
        network_save_params(n);
        t2 = time(NULL);
        printf("Saving took %ld seconds\n", t2 - t1); 
    }

    network_delete(n);
    config_delete(cfg);
    printf("Deleted network\n");

    if (reload) {
        t1 = time(NULL);
        printf("Allocating and loading network from saved parameters ...\n"); 
        cfg = config_new("./data/config.json");
        n = network_new(cfg, true);
        t2 = time(NULL);
        printf("Allocating and loading took %ld seconds\n", t2 - t1); 
        network_delete(n);
        config_delete(cfg);
        printf("Deleted network\n"); fflush(stdout);
    }
}


int main(int argc, char *argv[]) {
#if 0
    test_mt();
    test_norm();
    test_shuffle();
    test_uthash();
    test_wrnpy();
    test_dwtahash_lsht();
    test_layer(true);
    test_network(true, true);
#endif
    test_layer(true);
    return 0;
}
