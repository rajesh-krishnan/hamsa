SOURCES = mt19937/mt19937ar.c myhelper.c lsh.c dwtahash.c node.c layer.c
#network.c

CFLAGS  = -fPIC -fopenmp -shared -O3 -march=native
OBJECTS = $(SOURCES:.c=.o)

all: hamsa libhamsa.so

hamsa: main.c libhamsa.a
	gcc -o $@ -fopenmp $< libhamsa.a -lm
	strip $@

libhamsa.a: $(OBJECTS)
	ar r $@ $^

libhamsa.so: $(OBJECTS)
	gcc $(CFLAGS) -o $@ $^

clean:
	rm -f main libhamsa.* $(OBJECTS)

