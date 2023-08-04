#include "myhelper.h"
#include "cnpy/cnpy.h"

void *mymap(size_t size) {
    void *ptr;
    ptr = mmap(NULL, size, PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (ptr != MAP_FAILED) return ptr;
    fprintf(stderr, "hugetlb allocation failed %ld\n", size);
    ptr = mmap(NULL, size, PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr != MAP_FAILED) return ptr;
    fprintf(stderr, "mmap failed \n");
    exit(0);
}

void myunmap(void *ptr, size_t size) { munmap(ptr, size); }

void myrnginit() {
    static int inited = 0;
    int rd;
    unsigned long init[] = {0x123, 0x234, 0x345, 0x456};
    if (inited) return;
    if((rd = open("/dev/urandom", O_RDONLY)) < 0) {
        fprintf(stderr, "Could not open /dev/urandom\n");
    }
    else {
        if(read(rd, init, 4) < 0) fprintf(stderr, "Read from /dev/urandom failed\n");
    }
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

void save_fnpy(float *farr, bool twoD, size_t d0, size_t d1, char *fn) {
  cnpy_array a;
  size_t index[2];

  index[0] = d0;
  index[1] = (twoD ? d1 : 1);
  unlink(fn);
  if (cnpy_create(fn, CNPY_BE, CNPY_F4, CNPY_FORTRAN_ORDER, (twoD?2:1), index, &a) != CNPY_SUCCESS) {
    cnpy_perror("Unable to create file");
    abort();
  }

  float *t = farr;
  for (index[0] = 0; index[0] < d0; index[0]++) {
      for (index[1] = 0; index[1] < d1; index[1]++) {
          cnpy_set_f4(a, index, *t);
          t++;
      }
  }

  if (cnpy_close(&a) != CNPY_SUCCESS) {
    cnpy_perror("Unable to close file");
    abort();
  }
}

void load_fnpy(float *farr, bool twoD, size_t d0, size_t d1, char *fn) {
  cnpy_array a;
  size_t index[2];

  if (cnpy_open(fn, false, &a) != CNPY_SUCCESS) {
    cnpy_perror("Unable to load file");
    abort();
  }

  assert(a.n_dim == (twoD ? 2 : 1));
  assert(a.dims[0] == d0);
  if (a.n_dim==2) assert(a.dims[1] == d1);

  for (index[0] = 0; index[0] < d0; index[0]++) {
      for (index[1] = 0; index[1] < (twoD ? d1 : 1); index[1]++) {
          *farr = cnpy_get_f4(a, index);
          farr++;
      }
  }

  if (cnpy_close(&a) != CNPY_SUCCESS) {
    cnpy_perror("Unable to close file");
    abort();
  }
}

