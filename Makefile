SOURCES = sfmt/SFMT.c myhelper.c lsh.c dwtahash.c node.c layer.c
#network.c

OBJECTS  = $(SOURCES:.c=.o)
CC       = gcc
INCLUDES = -I . -I ./sfmt
CFLAGS   = -O3 -finline-functions -fno-strict-aliasing \
           --param max-inline-insns-single=1800 -Wall -std=c99 \
           -Wno-missing-prototypes -Wno-unused-variable -Wno-unused-function \
           -fPIC -finline-functions -fopenmp \
           -march=native -mtune=intel \
           -msse2 -DHAVE_SSE2 -DSFMT_MEXP=19937 -pg 

all: hamsa 

slib: libhamsa.so

hamsa: main.c libhamsa.a
	gcc $(INCLUDES) $(CFLAGS) -o $@ -fopenmp $< libhamsa.a -lm

libhamsa.a: $(OBJECTS)
	ar r $@ $^

libhamsa.so: $(OBJECTS)
	gcc $(INCLUDES) $(CFLAGS) -shared -o $@ $^

clean:
	rm -f hamsa libhamsa.* $(OBJECTS)

