SRC = mt19937/mt19937ar.c myhelper.c lsh.c dwtahash.c node.c 
#layer.c network.c

hamsa: main.c $(SRC)
	gcc -o $@ -fopenmp $^ -lm

clean:
	rm -f ./hamsa
