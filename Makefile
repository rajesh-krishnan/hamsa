SOURCES = sfmt/SFMT.c tiny-json/tiny-json.c npy_array/npy_array.c \
          myrand.c configfile.c \
          lsht.c dwtahash.c node.c layer.c network.c
OBJECTS  = $(SOURCES:.c=.o)
CFLAGS   = -O3 -finline-functions -fno-strict-aliasing \
           --param max-inline-insns-single=2000 -Wall -std=c99 \
           -Wno-missing-prototypes -Wno-unused-variable -Wno-unused-function \
           -fPIC -finline-functions -fopenmp \
           -march=native -mtune=intel \
           -msse2 -DHAVE_SSE2 -DSFMT_MEXP=19937 # -pg 
CPLFLAGS = -O3 -Wall -std=c++11 \
           -march=native -mtune=intel \
           -Wno-unused-variable -Wno-unused-function \
           -msse2 -DHAVE_SSE2 -DSFMT_MEXP=19937
CPP      = g++
CC       = gcc
AR       = ar
RANLIB   = ranlib

all: amazon640 test

amazon640: amazon640.cpp hamsalib
	$(CPP) $(CPLFLAGS) -o $@ -fopenmp $< libhamsa.a -lm

test: test.c hamsalib
	$(CC) $(CFLAGS) -o $@ -fopenmp $< libhamsa.a -lm

hamsalib: libhamsa.so libhamsa.a

libhamsa.a: $(OBJECTS)
	$(AR) rcs $@ $^
	$(RANLIB) $@

libhamsa.so: $(OBJECTS)
	$(CC) $(CFLAGS) -shared -o $@ $^

clean: libclean objclean execlean

objclean:
	rm -f $(OBJECTS)

libclean:
	rm -f libhamsa.a libhamsa.so 

execlean:
	rm -f test amazon640

