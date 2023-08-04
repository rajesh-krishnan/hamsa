#include "network.h"

void test_wrnpy() {
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

void test_mt() {
    int i,N = 10000;
    double totl;
    for (i=0, totl=0.0; i<N; i++) totl+=myrand_unif();
    printf("\nFor %d samples of myrand_unif() -- ", N);
    printf("Expected mean: %d  Observed mean: %d\n", INT_MAX/2, (int)(totl/N));
}

void test_norm() {
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

void test_myshuffle() {
    int i, x[100];
    for (i = 0; i < 100; i++) x[i] = i;
    myshuffle(x, 100);
    printf("\nOutput shuffle of [0:99]\n");
    for (i = 0; i < 100; i++) {
        printf("%d", x[i]);
        printf("%s", (i%20==19) ? "\n" : " ");
    }
}

void test_dwtahash() {
    printf("\nTesting Densified Winner Take All Hashing\n");
    DWTAHash *d = dwtahash_new(300, 128);
    dwtahash_delete(d);
}

void test_lsh(int K, int L, int R) {
    printf("\nTesting Locality Sensitive Hashing\n");
    LSH *l = lsh_new(K,L,R);
    lsh_clear(l);
    lsh_delete(l);
}

void test_layer(int io) {
    printf("\nTesting Layer \n");
    Layer *l = layer_new(670091, 128, 1, ReLU, 1024, 6, 50, 18, false, NULL);
    if (io) {
       layer_save(l, "/tmp");
       layer_load(l, "/tmp");
    }
    layer_delete(l);
}

void ht_update(khash_t(hist) *h) {
    int isnew;
    khiter_t k;
    for(size_t i = 2; i<10; i++) {
       k = kh_put(hist, h, i, &isnew);
       kh_value(h, k) = (isnew) ? 1 : (kh_value(h, k) + 1);
    }
    for(size_t i = 4; i<8; i++) {
       k = kh_put(hist, h, i, &isnew);
       kh_value(h, k) = (isnew) ? 1 : (kh_value(h, k) + 1);
    }
}

void test_hashtable() {
    khiter_t k;
    khash_t(hist) *h = kh_init(hist);
    ht_update(h);
    printf("num keys: %d\n", kh_size(h));
    for (k = kh_begin(h); k != kh_end(h); ++k)
       if (kh_exist(h, k)) printf("key=%d, value=%ld\n", k, kh_value(h, k));
    kh_destroy(hist, h);
}

int main(int argc, char *argv[]) {
/*
    test_hashtable();
    test_wrnpy();
    test_mt();
    test_norm();
    test_myshuffle();
    test_dwtahash();
    test_lsh(6,50,18);
*/
    test_norm();
    test_layer(0);
    return 0;
}
