#include <stdlib.h>
#include <stdio.h>
#include "bucket.h"
#include "mt19937/mt19937ar.h"

int main(int argc, char *argv[]) {
    int i;
    unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;

    Bucket * b = malloc(sizeof(Bucket));
    reset(b);
    add(b,3);
    printf("Size: %d\n", getSize(b));
    reset(b);
    printf("Size: %d\n", getSize(b));

    init_by_array(init, length);
    printf("10 outputs of genrand_int31()\n");
    for (i=0; i<10; i++) {
      printf("%10lu ", genrand_int32());
      if (i%5==4) printf("\n");
    }
    return 0;
}
