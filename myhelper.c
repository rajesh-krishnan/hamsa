#include "myhelper.h"

void *mymap (size_t size) {
    void *ptr;
    ptr = mmap(NULL, size, PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (ptr != MAP_FAILED) return ptr;
    fprintf(stderr, "hugetlb allocation failed %ld\n", size);
    ptr = mmap(NULL, size, PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr != MAP_FAILED) return ptr;
    fprintf(stderr, "mmap failed \n");
    exit(0);
}

void myunmap (void *ptr, size_t size) { munmap(ptr, size); }

void myrnginit() {
    static int inited = 0;
    int rd;
    unsigned long init[] = {0x123, 0x234, 0x345, 0x456};
    if (inited) return;
    if((rd = open("/dev/urandom", O_RDONLY)) >= 0) read(rd, init, 4);
    init_by_array(init, 4);
    inited = 1;
}

void myshuffle(int *array, int n) {
    if (n <= 1) return;
    for (int i = 0; i < n - 1; i++) {
        int j = i + (genrand_int31() % (n - i));
        assert(j >= 0 && j < n);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

float randnorm (double mu, double sigma) {
    double U1, U2, W, mult;
    static double X1, X2;
    static int call = 0;
 
    if (call) {
      call = !call;
      return (mu + sigma * X2);
    }
 
    do { 
        U1 = -1 + genrand_real1() * 2;
        U2 = -1 + genrand_real1() * 2;
        W = pow (U1, 2) + pow (U2, 2);
    } while (W >= 1 || W == 0);
 
    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;
    call = !call;
    return (float) (mu + sigma * X1);
}
