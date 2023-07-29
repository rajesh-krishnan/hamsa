#include "hdefs.h"
#include "bucket.h"
#include "lsh.h"
#include "mt19937/mt19937ar.h"

int main(int argc, char *argv[]) {
    int i;
    unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
  
    test_bucket();

    init_by_array(init, length);
    printf("10 outputs of genrand_int31()\n");
    for (i=0; i<10; i++) {
      printf("%10lu ", genrand_int32());
      if (i%5==4) printf("\n");
    }

    LSH *l = new_lsh(6, 50, 18);
    clear_lsh(l);
    delete_lsh(l);
    return 0;
}
