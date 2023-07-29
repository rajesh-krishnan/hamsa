#include "hdefs.h"
#include "bucket.h"
#include "lsh.h"
#include "mt19937/mt19937ar.h"

void test_bucket() {
    Bucket * b = malloc(sizeof(Bucket));
    reset(b);
    add(b,3);
    printf("Size: %d\n", b->count);
    printf("Retrieved: %d\n", retrieve(b,0));
    reset(b);
    printf("Size: %d\n", b->count);
}

void test_mt() {
    int i;
    unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
    init_by_array(init, length);
    printf("10 outputs of genrand_int31()\n");
    for (i=0; i<10; i++) {
        printf("%10lu ", genrand_int32());
        if (i%5==4) printf("\n");
    }
}

void test_lsh() {
    LSH *l = new_lsh(6, 50, 18);
    clear_lsh(l);
    delete_lsh(l);
}

int main(int argc, char *argv[]) {
    test_bucket();
    test_mt();
    test_lsh();
    return 0;
}
