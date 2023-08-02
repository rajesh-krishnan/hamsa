SOURCES = mt19937/mt19937ar.c myhelper.c lsh.c dwtahash.c node.c layer.c
#network.c

CFLAGS  = -fPIC -fopenmp -shared -O3 -march=native
OBJECTS = $(SOURCES:.c=.o)

all: hamsa

hamsa: main.c libs
	gcc -o $@ -fopenmp $< libhamsa.a -lm
	strip $@

libs: libhamsa.so libhamsa.a
	@rm -f $(OBJECTS)

libhamsa.so: $(OBJECTS)
	gcc $(CFLAGS) -o $@ $^

libhamsa.a: $(OBJECTS)
	ar r $@ $^

clean:
	rm -f main libhamsa.* $(OBJECTS)

