#include "hdefs.h"

int myrand_unif() { 
    static sfmt_t sfmt;
    static int inited = 0;
    if (inited) return(sfmt_genrand_uint32(&sfmt)>>1);
    int rd;
    unsigned int init[] = {0x123, 0x234, 0x345, 0x456};
    if((rd = open("/dev/urandom", O_RDONLY)) < 0) {
            fprintf(stderr, "Could not open /dev/urandom\n");
    }
    else {
            if(read(rd, init, 4) < 0) fprintf(stderr, "Read from /dev/urandom failed\n");
    }
    sfmt_init_by_array(&sfmt, init, 4);
    inited = 1;
    return(sfmt_genrand_uint32(&sfmt)>>1);
}

float myrand_norm(double mu, double sigma) { /* Box-Muller */
    static float scale = (1.0/0x7fffffff); 
    static float X1, X2;
    static int call = 0;
    float U1, U2;
    call = !call;
    if (!call) return X2;
    do { U1 = myrand_unif() * scale; } while (U1 == 0); /* Avoid log (0) */
    U1 = sqrtf(-2 * log(U1));
    U2 = 2 * MY_PI * myrand_unif() * scale;
    X1 = mu + U1 * cos(U2) * sigma;
    X2 = mu + U1 * sin(U2) * sigma;
    return X1;
}

void myrand_shuffle(int *array, int n) {
    if (n <= 1) return;
    for (int i = 0; i < n - 1; i++) {
        int j = i + (myrand_unif() % (n - i));
        assert(j >= 0 && j < n);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

