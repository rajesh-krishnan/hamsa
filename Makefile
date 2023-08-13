SOURCES = sfmt/SFMT.c tiny-json/tiny-json.c npy_array/npy_array.c \
          myrand.c configfile.c \
          lsht.c dwtahash.c node.c layer.c network.c
OBJECTS  = $(SOURCES:.c=.o)
CFLAGS   = -g -O3 -std=c99 \
           -Wall -Wno-missing-prototypes -Wno-unused-variable -Wno-unused-function \
           -finline-functions --param max-inline-insns-single=2000 -fno-strict-aliasing \
           -fPIC -fopenmp -ffast-math \
           -march=native -mtune=intel \
           -msse2 -DHAVE_SSE2 -DSFMT_MEXP=19937 
CPLFLAGS = -g -O3 -std=c++11 \
           -Wall -Wno-unused-variable -Wno-unused-function \
           -march=native -mtune=intel \
           -msse2 -DHAVE_SSE2 -DSFMT_MEXP=19937
CPP      = g++
CC       = gcc
AR       = ar
RANLIB   = ranlib

all: amazon640 test libhamsa.so

amazon640: amazon640.cpp libhamsa.a
	$(CPP) $(CPLFLAGS) -o $@ -fopenmp $< libhamsa.a -lm

test: test.c libhamsa.a
	$(CC) $(CFLAGS) -o $@ -fopenmp $< libhamsa.a -lm

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

