#include "network.h"

void test_savenpy() {
    float farr[20];
    printf("\nTest saving of 1D and 2D float arrays as npy files\n");
    for (int i=0; i<20; i++) farr[i] = (float) i;
    write_fnpy(farr, false, 20, 1, "float_20.npy");
    write_fnpy(farr, true, 1, 20, "float_1x20.npy");
    write_fnpy(farr, true, 20, 1, "float_20x1.npy");
    write_fnpy(farr, true, 4, 5, "float_4x5.npy");
}

void test_mt() {
    myrnginit();
    printf("\nOutput 20 draws of genrand_int31()\n");
    for (int i=0; i<20; i++) {
        printf("%10lu ", genrand_int31());
        if (i%10==9) printf("\n");
    }
}

void test_norm() {
    int i,N = 10000;
    float samp[10000];
    float totl, mean, stdv = 0;
    myrnginit();
    printf("\nDrawing %d items of randnorm()\n", N);
    for (i=0, totl=0.0; i<N; i++) {
        samp[i] = randnorm(0.0, 0.01);
        totl+=samp[i];
    }
    mean = totl / N;
    samp[i] = samp[i] - mean;
    for (i=0, totl=0.0; i<N; i++) totl += samp[i] * samp[i];
    stdv = sqrt(totl / N);
    printf("Expected mean,stdv: 0.0,0.01 Actual mean,stdv: %f,%f\n", mean, stdv);
}

void test_myshuffle() {
    int i, x[100];
    myrnginit();
    for (i = 0; i < 100; i++) x[i] = i;
    myshuffle(x, 100);
    printf("\nOutput shuffle of [0:99]\n");
    for (i = 0; i < 100; i++) {
        printf("%d", x[i]);
        printf("%s", (i%10==9) ? "\n" : " ");
    }
    printf("\n");
}

void test_bucket() {
    int i, *x;
    Bucket *b = malloc(sizeof(Bucket));
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
    free(b);
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
    Layer *l = layer_new(670091, 128, 1, ReLU, 1024, 6, 50, 18, 0.01, 1.0, false, NULL);
    layer_save(l, ".");
    layer_delete(l);
}

int main(int argc, char *argv[]) {
    test_savenpy();
    test_mt();
    test_norm();
    test_myshuffle();
    test_bucket();
    test_dwtahash();
    test_lsh(6,50,18);
    test_layer();
    return 0;
}
