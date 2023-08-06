#!/usr/bin/python3
from cffi import FFI

ffi = FFI()
with open('hamsa.h') as f: 
  ffi.cdef(f.read())
hamsa = ffi.dlopen('./libhamsa.so')

print('Building network from scratch');
cfg = hamsa.config_new(b'sampleconfig.json')
n   = hamsa.network_new(cfg, False);
print("Saving configuration and layer parameters");
hamsa.config_save(cfg, b'./data/config.json')
hamsa.network_save_params(n);
hamsa.network_delete(n);
hamsa.config_delete(cfg);

