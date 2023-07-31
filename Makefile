SRC = main.c mt19937/mt19937ar.c mhelper.c bucket.c lsh.c node.c dwtahash.c

hamsa: $(SRC)
	gcc -o $@ -fopenmp $^ -lm

clean:
	rm -f ./hamsa
