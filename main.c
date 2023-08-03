#include "network.h"

/*
Keep this until the counts capture code is tested
void test_bucket() {
    int i, *x;
    Bucket tmp;
    Bucket *b = &tmp;
    bucket_reset(b);
    bucket_add_to(b,3);
    bucket_add_to(b,2);
    bucket_add_to(b,4);
    bucket_add_to(b,9);
    printf("\nSize: %d In-buckets:\n", b->count);
    x = bucket_get_array(b);
    for (i = 0; i <= b->count; i++) {
        printf("%d", x[i]);
        printf("%s", (i%10==9) ? "\n" : " ");
    }
    printf("\n");
    bucket_reset(b);
    x = bucket_get_array(b);
    printf("\nSize: %d In-buckets:\n", b->count);
    for (i = 0; i <= b->count; i++) {
        printf("%d", x[i]);
        printf("%s", (i%10==9) ? "\n" : " ");
    }
    printf("\n");
}
*/

void test_wrnpy() {
    float ifarr[20];
    float ofarr[20];

    printf("\nTest saving and loading of 1D and 2D float arrays as npy files\n");
    for (int i=0; i<20; i++) ifarr[i] = i;

    save_fnpy(ifarr, false, 20, 1, "/tmp/float_20.npy");
    for (int i=0; i<20; i++) ofarr[i] = 0;
    load_fnpy(ofarr, false, 20, 1, "/tmp/float_20.npy");
    for (int i=0; i<20; i++) assert(ifarr[i] == ofarr[i]);

    save_fnpy(ifarr, true, 1, 20, "/tmp/float_1x20.npy");
    for (int i=0; i<20; i++) ofarr[i] = 0;
    load_fnpy(ofarr, true, 1, 20, "/tmp/float_1x20.npy");
    for (int i=0; i<20; i++) assert(ifarr[i] == ofarr[i]);

    save_fnpy(ifarr, true, 20, 1, "/tmp/float_20x1.npy");
    for (int i=0; i<20; i++) ofarr[i] = 0;
    load_fnpy(ofarr, true, 20, 1, "/tmp/float_20x1.npy");
    for (int i=0; i<20; i++) assert(ifarr[i] == ofarr[i]);

    save_fnpy(ifarr, true, 4, 5, "/tmp/float_4x5.npy");
    for (int i=0; i<20; i++) ofarr[i] = 0;
    load_fnpy(ofarr, true, 4, 5, "/tmp/float_4x5.npy");
    for (int i=0; i<20; i++) assert(ifarr[i] == ofarr[i]);
}

void test_mt() {
    int i,N = 10000;
    double totl;
    myrnginit();
    for (i=0, totl=0.0; i<N; i++) totl+=genrand_int31();
    printf("\nFor %d samples of genrand_int31() -- ", N);
    printf("Expected mean: %d  Observed mean: %d\n", INT_MAX/2, (int)(totl/N));
}

void test_norm() {
    int i,N = 10000;
    float samp[10000];
    float totl, mean, stdv = 0;
    myrnginit();
    printf("\nFor %d samples of randnorm() -- ", N);
    for (i=0, totl=0.0; i<N; i++) {
        samp[i] = randnorm(0.0, 0.01);
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
    myrnginit();
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
    printf("Get Random: %d\n", dwtahash_getRandDoubleHash(d, 10, 5));
    printf("Get Random: %d\n", dwtahash_getRandDoubleHash(d, 10, 5));
    dwtahash_delete(d);
}

void test_lsh(int K, int L, int R) {
    printf("\nTesting Locality Sensitive Hashing\n");
    LSH *l = lsh_new(K,L,R);
    lsh_clear(l);
    lsh_delete(l);
}

void test_layer() {
    printf("\nTesting Layer \n");
    Layer *l = layer_new(670091, 128, 1, ReLU, 1024, 6, 50, 18, false, NULL);
    layer_save(l, "/tmp");
    layer_load(l, "/tmp");
    layer_delete(l);
}

int main(int argc, char *argv[]) {
    test_wrnpy();
    test_mt();
    test_norm();
    test_myshuffle();
    test_dwtahash();
    test_lsh(6,50,18);
    test_layer();
    return 0;
}
