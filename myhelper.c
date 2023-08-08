#include "hdefs.h"
#include "sfmt/SFMT.h"
#include "npy_array/npy_array.h"

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

void mysave_fnpy(float *farr, size_t d0, size_t d1, char *fn) {
    npy_array_save( fn, NPY_ARRAY_BUILDER( farr, SHAPE( d0, d1 ), NPY_DTYPE_FLOAT32 ) );
}

void myload_fnpy(float *farr, size_t d0, size_t d1, char *fn) {
    npy_array_t *a = npy_array_load(fn);
    assert(a != NULL);
    assert((a->typechar == 'f') && (a->elem_size == sizeof(float)) && (a->ndim == 2));
    assert(a->shape[0] == d0);
    assert(a->shape[1] == d1);
    memcpy(farr, a->data, d0 * d1 * sizeof(float));
    npy_array_free(a);
}

inline void __attribute__((always_inline)) ht_put(Histo **counts, int key, size_t value) {
    Histo *s;
    HASH_FIND_INT(*counts, &key, s);  
    if (s == NULL) {
        s = (Histo *)malloc(sizeof *s);
        s->key = key;
        HASH_ADD_INT(*counts, key, s);
    } 
    s->value = value;
}

inline void __attribute__((always_inline)) ht_incr(Histo **counts, int key) {
    Histo *s;
    HASH_FIND_INT(*counts, &key, s);  
    if (s == NULL) {
        s = (Histo *)malloc(sizeof *s);
        s->key = key;
        HASH_ADD_INT(*counts, key, s);
        s->value = 1;
    } else {
        s->value += 1;
    }
}

inline void __attribute__((always_inline)) ht_delkey(Histo **counts, int key) {
    Histo *s;
    HASH_FIND_INT(*counts, &key, s);  
    if (s != NULL) {
        HASH_DEL(*counts, s);
        free(s);
    }
}

inline void __attribute__((always_inline)) ht_del(Histo **counts, Histo **cur) {
    HASH_DEL(*counts, *cur); 
    free(*cur);
}

inline void __attribute__((always_inline)) ht_destroy(Histo **counts) {
    Histo *cur, *tmp;
    HASH_ITER(hh, *counts, cur, tmp) {
        HASH_DEL(*counts, cur);
        free(cur);
    }
}

inline unsigned int __attribute__((always_inline)) ht_size(Histo **counts) { return HASH_COUNT(*counts); }

