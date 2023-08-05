# HAMSA: Hash Aided Machine-intelligence Sans Accelerators

This project seeks to enable machine intelligence based on neural networks
aided by hash algorithms without requiring hardware accelerators such as
Graphics Processor Unit (GPU), Field Programmable Gate Array (FPGA), or
neuromorphic computing architectures. A high-end x86 server with solid state
disk (SSD) and adequate random access memory (RAM) running Linux with HugeTLB
Pages enabled, with gcc compiler and OpenMP support should suffice. Our development
servers 

This implementation follows the SLIDE paper referenced below and the author's
original implementation in C++.  Our implementation is in C with minimal external 
dependencies; included third-party dependencies include klib, tiny-json, SFMT,
and cnpy.  We have refactored the code considerably, included code to replace 
C++ std functions, eliminated code repetition, and dependency on C++ cnpy and 
zlib. We import/export to `.npy` files instead of the `.npz` archives. Based
on the results reported in the paper, we have chosen to implement only the
Densified Winner Take All Hash, First-In-First-Out sampling, and Adaptive
Moment Estimation gradient descent. Either thresholding or minimum active nodes
per layer can be chosen at compilation. We reduce the number of places dynamic
memory allocation is done, and furthermore we use mmap for the larger
allocations systematically (in layers and lsh).  Instead of the custom config 
file, we use JSON from which the network can be loaded; the config can also be
saved.  We also build a shared library to allow calling from Python.

Once the core capability is built and tested, we plan to focus on two problem
domains: (i) natural language processing using large language models and (ii)
stochatic planning on continuous domains using RDDL.

## References
   1. Beidi Chen, Tharun Medini, James Farwell, Gobrial Sameh, Charlie Tai, and Anshumali Shrivastava, "SLIDE : In Defense of Smart Algorithms over Hardware Acceleration for Large Scale Deep Learning Systems," MLSys 2020
   2. Wu Ga, Buser Say, and Scott Sanner, "Scalable Planning with Tensorflow for Hybrid Nonlinear Domains," NeurIPS 2017
