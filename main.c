#include "network.h"

void test_mt() {
    myrnginit();
    printf("\nOutput 20 draws of genrand_int31()\n");
    for (int i=0; i<20; i++) {
        printf("%10lu ", genrand_int31());
        if (i%10==9) printf("\n");
    }
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

void test_lsh(int K, int L, int R) {
    printf("\nTesting Locality Sensitive Hashing\n");
    LSH *l = lsh_new(K,L,R);
    lsh_clear(l);
    lsh_delete(l);
}

int main(int argc, char *argv[]) {
    test_mt();
    test_myshuffle();
    test_bucket();
    test_lsh(6,50,18);
    return 0;
}
