SOURCES = mt19937/mt19937ar.c myhelper.c lsh.c dwtahash.c node.c layer.c
#network.c

CFLAGS  = -fPIC -fopenmp -shared -O3 -march=native 
OBJECTS = $(SOURCES:.c=.o)

all: hamsa 

slib: libhamsa.so

hamsa: main.c libhamsa.a
	gcc -o $@ -fopenmp $< libhamsa.a -lm -lomp

libhamsa.a: $(OBJECTS)
	ar r $@ $^

libhamsa.so: $(OBJECTS)
	gcc $(CFLAGS) -o $@ $^

clean:
	rm -f hamsa libhamsa.* $(OBJECTS)

