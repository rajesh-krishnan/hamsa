#!/usr/bin/python3
from cffi import FFI
import timeit

ffi = FFI()
with open('hamsa.h') as f: 
  ffi.cdef(f.read())
hamsa = ffi.dlopen('./libhamsa.so')

print('Building network from scratch');
cfg = hamsa.config_new(b'sampleconfig.json')

start_time = timeit.default_timer()
n   = hamsa.network_new(cfg, False);
elapsed = timeit.default_timer() - start_time
print(elapsed)

print("Saving configuration and layer parameters");
hamsa.config_save(cfg, b'./data/config.json')
hamsa.network_save_params(n);
hamsa.network_delete(n);
hamsa.config_delete(cfg);

