#!/usr/bin/python3
from cffi import FFI
from time import time
from datetime import datetime

with open('hamsa.h') as f: 
    ffi = FFI()
    ffi.cdef(f.read())
    hamsa = ffi.dlopen('./libhamsa.so')

# Amazon640 BOW file format is:
#    header-line: nrecords nfeatures nclasses
#    record-line: label,..,label feature:value feature:value ... feature:value
def procLn(line):
    toks = line.split(' ')
    f = {int(x[0]):float(x[1]) for x in [y.split(':') for y in toks[1:]]}
    l = [int(x) for x in toks[0].split(',')]
    return (ffi.new('int[]',list(f.keys())), ffi.new('float[]',list(f.values())), len(f), 
            ffi.new('int[]',list(l)), len(l))

def NextBatchData(fname, offset, batchsize):
    with open(fname, 'r') as f:
        if offset == 0: f.readline()                                # skip header
        else:           f.seek(offset, 0);                          # seek to start position for this batch
        d = zip(*[procLn(f.readline()) for _ in range(batchsize)])  # process rows and transpose matrix
        batch = list(list(x) for x in d)                            # convert to list of lists vs. generator
        offset = f.tell()                                           # save start position for next batch
    records,values,sizes,labels,labelsize = batch
    return (ffi.new('int *[]',records), ffi.new('float *[]',values), ffi.new('int[]',sizes),
            ffi.new('int *[]',labels), ffi.new('int[]',labelsize), offset, batch) # batch avoids garbage collection

def doTest(net, sample=False):
    offset     = 0
    datafile   = ffi.string(net._cfg.testData)
    batchsize  = int(net._cfg.Batchsize)
    numBatches = int(net._cfg.totRecordsTest/batchsize)
    totCorrect = 0
    numEval    = 20 if sample else numBatches
    print('Start evaluation on test data at %s' % datetime.now())
    for i in range(numEval):
        records,values,sizes,labels,labelsize,offset,ka = NextBatchData(datafile, offset, batchsize)
        totCorrect += hamsa.network_infer(net, records, values, sizes, labels, labelsize);
    accuracy = totCorrect * 1.0 / (numEval*batchsize)
    print('Finish evaluation on test data at %s' % datetime.now())
    return accuracy

def doTrain(net):
    datafile   = ffi.string(net._cfg.trainData)
    logfile    = ffi.string(net._cfg.logFile)
    batchsize  = int(net._cfg.Batchsize)
    stepsize   = int(net._cfg.Stepsize)
    numBatches = int(net._cfg.totRecords/batchsize)
    nRehash    = int(net._cfg.Rehash/batchsize)
    nRebuild   = int(net._cfg.Rebuild/batchsize)
    nReperm    = int(net._cfg.Reperm/batchsize)
    trainTime  = 0
    perf       = 0

    for epoch in range(cfg.Epoch):
        print('Start training epoch %d at %s' % (epoch,datetime.now()))
        offset = 0
        for i in range(numBatches):
            cbatchnum = epoch * numBatches + i

            if ((cbatchnum % stepsize) == 0): 
                perf = doTest(n, True)
                with open(logfile, 'a') as lf:
                    print('PROGRESS: Epoch %d Batches %d Training %.4f s Accuracy %0.4f' % 
                          (epoch, cbatchnum, trainTime, perf), file=lf)

            records,values,sizes,labels,labelsize,offset,ka = NextBatchData(datafile, offset, batchsize)
            rehash  = ((cbatchnum % nRehash)  == (nRehash-1))
            rebuild = ((cbatchnum % nRebuild) == (nRebuild-1))
            reperm  = ((cbatchnum % nReperm)  == (nReperm-1))

            t = time()
            hamsa.network_train(net, records, values, sizes, labels, labelsize, cbatchnum, reperm, rehash, rebuild);
            trainTime += (time() - t)

            if ((cbatchnum % 100) == 99): print('Trained batches %d in %.4f seconds' % (cbatchnum+1, trainTime))

        print('Finish training epoch %d at %s' % (epoch,datetime.now()))
        hamsa.network_save_params(n);

if __name__ == '__main__':
    inCfgFile  = b'sampleconfig.json'
    savCfgFile = b'./data/config.json'

    cfg = hamsa.config_new(b'sampleconfig.json')
    print('Loaded config')

    start_time = time()
    n = hamsa.network_new(cfg, False);
    elapsed = time() - start_time
    print('Network initialization takes %.4f seconds' % elapsed)

    hamsa.config_save(cfg, savCfgFile);
    print('Saved network configuration')
    hamsa.network_save_params(n);

    logfile = ffi.string(cfg.logFile)
    with open(logfile, 'w') as lf: print('Opening log file at %s' % datetime.now(), file=lf)
    doTrain(n)
    acc = doTest(n)
    with open(logfile, 'a') as lf: print('Accuracy on test data %.4f at %s' % (acc,datetime.now()), file=lf)

    hamsa.network_delete(n);
    hamsa.config_delete(cfg);
    print('Done')

