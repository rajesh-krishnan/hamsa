SRC = main.c mt19937/mt19937ar.c mhelper.c bucket.c node.c lsh.c

hamsa: $(SRC)
	gcc -o $@ -fopenmp $^ 

clean:
	rm -f ./hamsa
