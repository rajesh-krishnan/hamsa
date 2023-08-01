# HAMSA: Hash Aided Machine-intelligence Sans Accelerators

This project seeks to enable machine intelligence based on neural networks aided by hash algorithms without requiring hardware accelerators such as Graphics Processor Unit (GPU), Field Programmable Gate Array (FPGA), or neuromorphic computing architectures. A high-end x86 server with solid state disk (SSD) and adequate random access memory (RAM) should suffice.  We plan to have an implementation in C with python bindings, and with minimal external dependencies.

We plan to focus on two problem domains: (i) natural language processing using large language models of up to 20B parameters and (ii) stochatic planning on continuous domains using RDDL.

From the SLIDE paper, we implement only the Densified Winner Take All Hash, FIFO sampling, and ADAM SGD. We reduce 
dynamic memory allocation where possible.

## References
   1. Beidi Chen, Tharun Medini, James Farwell, Gobrial Sameh, Charlie Tai, and Anshumali Shrivastava, "SLIDE : In Defense of Smart Algorithms over Hardware Acceleration for Large Scale Deep Learning Systems," MLSys 2020
   2. Wu Ga, Buser Say, and Scott Sanner, "Scalable Planning with Tensorflow for Hybrid Nonlinear Domains," NeurIPS 2017
