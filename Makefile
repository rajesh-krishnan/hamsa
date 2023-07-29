SRC = main.c mhelper.c bucket.c mt19937/mt19937ar.c

hamsa: $(SRC)
	gcc -o $@ $^ 
