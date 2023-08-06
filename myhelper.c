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

void myshuffle(int *array, int n) {
    if (n <= 1) return;
    for (int i = 0; i < n - 1; i++) {
        int j = i + (myrand_unif() % (n - i));
        assert(j >= 0 && j < n);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

inline float __attribute__((always_inline)) myrand_norm(double mu, double sigma) { /* Box-Muller */
    static double scale = (1.0/0x7fffffff); 
    static double X1, X2;
    static int call = 0;
    double U1, U2;
    call = !call;
    if (!call) return (float) X2;
    do { U1 = myrand_unif() * scale; } while (U1 == 0); /* Avoid log (0) */
    U1 = sqrt(-2 * log(U1));
    U2 = 2 * MY_PI * myrand_unif() * scale;
    X1 = mu + U1 * cos(U2) * sigma;
    X2 = mu + U1 * sin(U2) * sigma;
    return (float) X1;
}

inline int __attribute__((always_inline)) myrand_unif() { 
    static sfmt_t sfmt;
    static int inited = 0;
    int rd;
    unsigned int init[] = {0x123, 0x234, 0x345, 0x456};
    if (!inited) {
        if((rd = open("/dev/urandom", O_RDONLY)) < 0) {
            fprintf(stderr, "Could not open /dev/urandom\n");
        }
        else {
            if(read(rd, init, 4) < 0) fprintf(stderr, "Read from /dev/urandom failed\n");
        }
        sfmt_init_by_array(&sfmt, init, 4);
        inited = 1;
    }
    return(sfmt_genrand_uint32(&sfmt)>>1);
}

void mysave_fnpy(float *farr, bool twoD, size_t d0, size_t d1, char *fn) {
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

void myload_fnpy(float *farr, bool twoD, size_t d0, size_t d1, char *fn) {
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

