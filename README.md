# HAMSA: Hash Aided Machine-intelligence Sans Accelerators

## Scope and Purpose

This project seeks to enable machine intelligence based on neural networks
aided by hash algorithms without requiring hardware accelerators such as
Graphics Processor Unit (GPU), Field Programmable Gate Array (FPGA), or
neuromorphic computing architectures. An x86-64 server with solid state
disk (SSD) and adequate random access memory (RAM) running Linux with 
HugeTLB Pages enabled, with gcc compiler and OpenMP should suffice.

## Implementation Details and Limitations

We provide a basic Makefile, which you will need to adapt to your needs.

An example usage following the original code is `amazon640`. Additionally,
the `test` program allows testing components. An example of how to use
the library from Python is provided in `test.py`.

The configuration I use for testing is below:

```
$ cat /proc/meminfo | grep MemTotal
MemTotal:       395004008 kB
DirectMap1G:    377487360 kB

$cat /proc/cpuinfo | grep processor | wc -l
32

$ cat /proc/cpuinfo | grep "model name" | uniq
model name	: Intel(R) Xeon(R) CPU E5-2667 v3 @ 3.20GHz

$ cat /sys/kernel/mm/transparent_hugepage/enabled
always [madvise] never

$ sudo hugeadm --pool-pages-max 2MB:60000
$ sudo hugeadm --pool-pages-min 2MB:10000

$ hugeadm --pool-list
      Size  Minimum  Current  Maximum  Default
   2097152    10000    10000    60000        *
1073741824        0        0        0         
```

Our initial implementation is based directly on the SLIDE paper [Chen2020](#Chen2020) and 
the author's original [implementation](https://github.com/RUSH-LAB/SLIDE) in C++. 
We have refactored the code considerably:

  * Our implementation is in C99, and does not require std-c++11, zlib, or C++
    cnpy. We import/export to `.npy` files instead of the `.npz` archives. We
    include source for third-party dependencies, namely, SIMD-Focused Mersenne
    Twister for random number genration, uthash for hashtables, tiny-json for
    parsing JSON, and npy_array to load and save NPY files. All the NPY files
    are formatted as 2D arrays for storage (the original uses 1D for bias and 
    2D for weights).

  * Our code is released under the MIT License, and dependencies use either the
    MIT License or a BSD-like License.

  * Based on the results reported in the paper, we have chosen to implement
    only the Densified Winner Take All Hash, First-In-First-Out sampling, and
    Adaptive Moment Estimation gradient descent. Either thresholding or minimum
    active nodes per layer can be chosen at compilation.  We have eliminated
    code repetition, fixed several bugs, improv

  * We use the SFMT library built with SSE2 and AVX256 support.

  * Currently omp parallelism is used in only 3 places:
    -- forward propagation in parallel across inputIDs in a batch for inference
    -- forward and back in parallel across inputIDs in a batch for training
    -- gradient descent in parallel across nodes of a layer, after forward and
       back propagation across all inputIDs in a a batch

  * Our refactor has eliminated a few places where a variable could be modified
    by multiple threads in parallel.

  * Instead of the custom config file, we use JSON from which the network can 
    be loaded; the config can also be saved.  

  * We have fixed a couple of bugs in the Densified Winner Take All hash that
    were causing memory violations when there is densification failure. Also
    the code was incorrectly using log() instead of log2()

  * We have reduced the number of dynamic memory allocations, in favor of a 
    single bulk mmap/munmap in Layer and LSHT. The Node struct is just convenience
    pointers with all node state allocated in Layer. This also allows clean saving
    and loading of parameters and optimization state from Layer. 

  * We have tried to keep memory allocation and freeing the reponsibility of 
    the caller and provide paired _new/_delete calls for config, network, layer,
    lsht, and dwtahash. Usually it is easy to match up the allocation and freeing.
    Special care in needed with some of the internal functions, for example,
    dwtahash_getHashEasy and dwtahash_getHash allocate and return an array for 
    hashes, which must be freed by the caller. Similarly ht_incr and ht_put 
    allocate memory for a hashtable entry which must be freed by ht_delkey or
    ht_del; fortunately there is ht_destroy, which will free memory associated 
    with all entries in the hashtable. There are also functions (dupints, 
    dupfloats, and dupstr) in the JSON parsing code which allocate memory,
    which is freed later in config_delete. The hashtable and JSON parsing code
    are awkward in C.

  * We have reduced code repetition in many places, and used either inlined
    functions or macros.

  * We use Kaiming initialization instead of the arbitrary N(0,0.01) in the
    code for weights and biases. We initialize bias to zero.  We systematically 
    save both parameters and optimizations state. The original code had a copy 
    of bias in node and another in layer, and the wrong variable was saved. 
    Likewise, ADAM biases were not being saved. Now it is possible to checkpoint
    training state and resume cleanly.

  * We also build a shared library to allow calling from Python, and provide 
    an example.

  * We have added a number of assertions across the code. 

  * For now, we have kept the main.cpp, renamed to amazon640.cpp, in C++. 
    We have refactored it to call our C implementation instead, and reduced
    some code repetition. 

### Notes on Configuration

Neural network configuration seems to be largely a black art. However, some notes
on the configuration file may help, which is particualr for the Amazon640K bag of 
words dataset.

`BETA1` and `BETA2` defined in `hdefs.h` are used to compute the temproary learning 
rate and ADAM parameters. `EPS` is a small constant used to avoid divide by zero issues.
`BUCKETSIZE` and `BINSIZE` are used for configuring hashing in conjunction with
other parameters as described below.

`MINACTIVE` and `THRESH` determine sampling strategy of active nodes when using locality
sensitive hashing. One of them must be zero.  If `THRESH > 0`, only those nodes retrieved
from the hash which have a count of `THRESH` are more are considered. If `MINACTIVE > 0`
additional nodes are randomly added to the selection to the minimum of the layer size
or `MINACTIVE`.


```
"InputDim":        135909
"numLayer":        2
"sizesOfLayers":   [128,670091]
"layersTypes":     [1,2]
```

The input "layer" has 135909 nodes. Each input record is a feature-vector of
135909 positive floats; however as the input vector is sparse (i.e., mostly
elements are zero), it is specified as a feature-index:feature-value pair if
the feature-value is greater than 0.  The input has 670091 possible labels
(output), and associated with each input record (in test and train data) is one
or more of these labels.

We configure two hidden layers, with the first being a Rectified Linear Unit
(ReLU) layer with 128 nodes.  The second layer is a Softmax with 670091 nodes
i.e., number of possible labels for classification. The two layers when
instantiated will have as the previous layer number of nodes as 135909 and 128.

```
"K":               [2,6],
```
The number of hashes, it is unclear how K is selected for a given scenario.

```
"RangePow":        [6,18],
```
Note `BINSIZE` defined as 8 in `hdefs.h`. The algorithm requires that  
`K * floor(log2(BINSIZE)) == RangePow`, which explains the choices. Each
LSHT will have `L* (1<<RangePow)` buckets, with each bucket having
`BUCKETSIZE` elements where `BUCKETSIZE` is defiens to be 128 in `hdefs.c`.

```
"L":               [20,50],
```
It is unclear how L should be chosen (perhaps the paper explains this),
but note that `K*L` is number of DWTA hashes for the layers.

```
"Sparsity":        [1,0.005,1,1],
```
This parameter is misleading. If only matters to the code whether 
`Sparsity == 1` or `Sparsity < 1`. In the latter case, locality 
sensitive hashing is used to select active nodes, otherwise all
nodes in the layer are considered.  `Sparsity > 1` is undefined.
The length of the array is twice the number of layers. The first 
half of the array is used for training the corresponding layer,
and the second half is used for inference on the corresponding 
layer. 

```
"trainData":       "./data/Amazon670K.bow/train.txt",
"totRecords":      490449,
```
The training data file and number of records in it
are specified. Current code assumes Amazon640K BOW data 
and format.

```
"testData":        "./data/Amazon670K.bow/test.txt",
"totRecordsTest":  153025,
```
The test data file and number of records in it
are specified. Current code assumes Amazon640K BOW data 
and format.

```
"Epoch":           10,
```
This is the number of training passes made through the entire 
training data set.

```
"Batchsize":       128,
```
This is a batch for training, i.e., this is the number of records 
to do forward propagation and back propagation on before running 
gradient descent.

```
"Lr":              0.0001,
```
Initial learning rate, temporary learning rate is updated every 
batch using a formula within the code.

```
"Stepsize":        1000,
```
The code checks for progress every Stepsize batches by running inference 
on a small number of batches of test data. 

```
"Rehash":          128000,
```
Once every Rehash/Bathsize batches, the LSHT counts are cleared and 
nodes/weights are rehashed into the LSHT. It is unclear how to choose this
setting.

```
"Rebuild":         128000,
```
Once every Rebuild/Bathsize batches, delete and get a new DWTA hasher.
It is unclear how to choose this setting.

```
"Reperm":         640000,
```
Once every Reperm/Bathsize batches, repermute the randNodes list.
It applies only to the last layer, and it is unclear how to choose 
this setting. The original code had this hardcoded to `6946*Batchsize`
and I chose a "nicer" multiple of the Batchsize.

```
"loadPath":        "./data",
```
The directory from which parameters and optimization state is loaded for 
each layer. These are stored as 2D arrays in Numpy file format.

```
"savePath":        "./data",
```
The directory to which parameters and optimization state are saved for 
each layer. These are stored as 2D arrays in Numpy file format.

```
"logFile":         "./data/log.txt"
```
Logs the learning progress for each epoch every Stepsize batches, 
recording the number of batches trained, the total time spent in
`network_train`, and the fraction of correct predictions.

### Caveats

The implementation is research-grade code. Some bugs may have been inadvertently
introduced in the refactor. The code requires further checks for correctness, 
signedness and overflows, corner cases, and concurrency.

## Future Work

Once the core capability is built and tested, we plan to focus on two problem
domains: (i) natural language processing using large language models and (ii)
stochatic planning on continuous domains using RDDL (see [Ga2017](#Ga2017)).

Code improvements that would help:
  * Use a logging function with timestamp, uniform format, and level
  * Replace amazon640.cpp with Python or C
  * Check whether amazon640 data can be redistributed -- was hard to find it

## References
   1. <a name=Chen2020></a>Beidi Chen, Tharun Medini, James Farwell, Gobrial Sameh, Charlie Tai, and
      Anshumali Shrivastava, "SLIDE : In Defense of Smart Algorithms over Hardware
      Acceleration for Large Scale Deep Learning Systems," MLSys 2020
   2. <a name=Ga2017></a>Wu Ga, Buser Say, and Scott Sanner, "Scalable Planning with Tensorflow
      for Hybrid Nonlinear Domains," NeurIPS 2017
