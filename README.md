# HAMSA: Hash Aided Machine-intelligence Sans Accelerators

This project seeks to enable machine intelligence based on neural networks
aided by hash algorithms without requiring hardware accelerators such as
Graphics Processor Unit (GPU), Field Programmable Gate Array (FPGA), or
neuromorphic computing architectures. A high-end x86 server with solid state
disk (SSD) and adequate random access memory (RAM) running Linux with HugeTLB
Pages enabled should suffice.  

This implementation in C with minimal dependencies is based on the SLIDE paper 
referenced below. We have implemented only the Densified Winner Take All hash, 
first-in-first-out sampling, and Adaptive Moment Estimation gradient descent.
We reduce dynamic memory allocation where possible. We import/export to `.npy`
files and not the `.npz` archives (so zlib and C++ cnpy are not needed).

In the future, we will explore: (i) speeding up random number genration usinf
the SIMD Focused Mersenne Twister (SFTP) implementation; (ii) speeding up
vector operations using AVX instructions; and (iii) speeding up processing
through OpenMP multi-processor parallelism.

Once the core capability is built and tested, we plan to focus on two problem
domains: (i) natural language processing using large language models and (ii)
stochatic planning on continuous domains using RDDL.

## References
   1. Beidi Chen, Tharun Medini, James Farwell, Gobrial Sameh, Charlie Tai, and Anshumali Shrivastava, "SLIDE : In Defense of Smart Algorithms over Hardware Acceleration for Large Scale Deep Learning Systems," MLSys 2020
   2. Wu Ga, Buser Say, and Scott Sanner, "Scalable Planning with Tensorflow for Hybrid Nonlinear Domains," NeurIPS 2017
