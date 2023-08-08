SOURCES = sfmt/SFMT.c tiny-json/tiny-json.c npy_array/npy_array.c \
          myhelper.c configfile.c \
          lsht.c dwtahash.c node.c layer.c network.c

OBJECTS  = $(SOURCES:.c=.o)
CC       = gcc
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
	ar rcs $@ $^
	ranlib $@

libhamsa.so: $(OBJECTS)
	gcc $(INCLUDES) $(CFLAGS) -shared -o $@ $^

clean:
	rm -f hamsa libhamsa.a libhamsa.so $(OBJECTS)

