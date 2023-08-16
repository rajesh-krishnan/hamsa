#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

extern "C" {
#include "hamsa.h"
}

using namespace std;

#define ALLOC_MEMORY_LOAD_NEXTBATCH \
        int **records = new int *[Batchsize]; \
        float **values = new float *[Batchsize]; \
        int *sizes = new int[Batchsize]; \
        int **labels = new int *[Batchsize]; \
        int *labelsize = new int[Batchsize]; \
        int nonzeros = 0; \
        int count = 0; \
        vector<string> list; \
        vector<string> value; \
        vector<string> label; \
        while (std::getline(file, str)) { \
            char *mystring = &str[0]; \
            char *pch, *pchlabel; \
            int track = 0; \
            list.clear(); \
            value.clear(); \
            label.clear(); \
            pch = strtok(mystring, " "); \
            pch = strtok(NULL, " :"); \
            while (pch != NULL) { \
                if (track % 2 == 0) \
                    list.push_back(pch); \
                else if (track%2==1) \
                    value.push_back(pch); \
                track++; \
                pch = strtok(NULL, " :"); \
            } \
            pchlabel = strtok(mystring, ","); \
            while (pchlabel != NULL) { \
                label.push_back(pchlabel); \
                pchlabel = strtok(NULL, ","); \
             } \
            nonzeros += list.size(); \
            records[count] = new int[list.size()]; \
            values[count] = new float[list.size()]; \
            labels[count] = new int[label.size()]; \
            sizes[count] = list.size(); \
            labelsize[count] = label.size(); \
            int currcount = 0; \
            vector<string>::iterator it; \
            for (it = list.begin(); it < list.end(); it++) { \
                records[count][currcount] = stoi(*it); \
                currcount++; \
            } \
            currcount = 0; \
            for (it = value.begin(); it < value.end(); it++) { \
                values[count][currcount] = stof(*it); \
                currcount++; \
            } \
            currcount = 0; \
            for (it = label.begin(); it < label.end(); it++) { \
                labels[count][currcount] = stoi(*it); \
                currcount++; \
            } \
            count++; \
            if (count >= Batchsize) \
                break; \
        }

#define RELEASE_MEMORY \
        delete[] sizes; \
        delete[] labels; \
        for (int d = 0; d < Batchsize; d++) { \
            delete[] records[d]; \
            delete[] values[d]; \
        } \
        delete[] records; \
        delete[] values;

float globalTime = 0;

void EvalDataSVM(Config *cfg, Network *mynet, int numBatchesTest, int iter) {
    int Batchsize = cfg->Batchsize;
    int totCorrect = 0;
    ofstream outputFile(cfg->logFile,  std::ios_base::app);
    ifstream file(cfg->testData);
    string str;
    getline(file, str); //Skip header

    for (int i = 0; i < numBatchesTest; i++) {
        ALLOC_MEMORY_LOAD_NEXTBATCH
        auto correctPredict = network_infer(mynet, records, values, sizes, labels, labelsize);
        totCorrect += correctPredict;
        RELEASE_MEMORY
    }
    file.close();
    cout << "Progress: " << iter << " " << globalTime/1000 << " " << totCorrect * 1.0 / (numBatchesTest*Batchsize) << endl;
    outputFile << iter << " " << globalTime/1000 << " " << totCorrect * 1.0 / (numBatchesTest*Batchsize) << endl;
    outputFile.close();
}

void ReadDataSVM(Config *cfg, Network* mynet, int numBatches, int epoch){
    int Batchsize = cfg->Batchsize;
    int Rehash = cfg->Rehash;
    int Rebuild = cfg->Rebuild;
    int Reperm = cfg->Reperm;
    int Stepsize = cfg->Stepsize;

    ifstream file(cfg->trainData);
    string str;
    getline( file, str ); // Skip header
    for (int i = 0; i < numBatches; i++) {
        ALLOC_MEMORY_LOAD_NEXTBATCH

        if((i+epoch*numBatches)%Stepsize==0) {
            cout << "Checking progress on batch " << (i+epoch*numBatches) << " at time " << time(NULL) << endl;
            EvalDataSVM(cfg, mynet, 10, epoch*numBatches+i);
            cout << "Logged progress on batch " << (i+epoch*numBatches) << " at time " << time(NULL) << endl;
        }

        bool rehash  = ((epoch*numBatches+i)%(Rehash/Batchsize) == (Rehash/Batchsize-1));
        bool rebuild = ((epoch*numBatches+i)%(Rebuild/Batchsize) == (Rebuild/Batchsize-1));
        bool reperm  = ((epoch*numBatches+i)%(Reperm/Batchsize) == (Reperm/Batchsize-1));

        auto t1 = std::chrono::high_resolution_clock::now();
        network_train(mynet, records, values, sizes, labels, labelsize, epoch * numBatches + i,
            reperm, rehash, rebuild);
        auto t2 = std::chrono::high_resolution_clock::now();
        int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        globalTime+= timeDiffInMiliseconds;

        RELEASE_MEMORY
        if((i+epoch*numBatches)%100==99) {
            cout << "Trained on " << (i+epoch*numBatches) + 1 << " batches at " << time(NULL) << " s " << endl;
        }
    }
    file.close();
}

int main(int argc, char* argv[]) {
    const char *inCfgFile  = "sampleconfig.json";
    const char *savCfgFile = "./data/config.json";
    Config *cfg = config_new(inCfgFile);
    cout << "Loaded config" << endl;

    ofstream f(cfg->logFile, std::ios_base::out); f.close();

    auto t1 = std::chrono::high_resolution_clock::now();
    Network *n = network_new(cfg, false);
    auto t2 = std::chrono::high_resolution_clock::now();
    float timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    cout << "Network initialization takes " << timeDiffInMiliseconds/1000 << " milliseconds" << endl;

    config_save(cfg, savCfgFile);
    cout << "Saved network configuration" << endl;
    network_save_params(n);

    int numBatches = cfg->totRecords / cfg->Batchsize;
    int numBatchesTest = cfg->totRecordsTest / cfg->Batchsize;
    int e = 0;
    while(e < cfg->Epoch) {
        ofstream outputFile(cfg->logFile, std::ios_base::app);
        outputFile<<"Epoch "<<e<<endl;
        outputFile.close();
        cout << "Start training epoch " << e << " at " << time(NULL) << " s " << endl;
        ReadDataSVM(cfg, n, numBatches, e);
        cout << "Completed training epoch " << e << " at " << time(NULL) << " s " << endl;
        network_save_params(n);
        e++;
    }
    ofstream outputFile(cfg->logFile, std::ios_base::app);
    outputFile<<"Evaluation"<<endl;
    outputFile.close();
    cout << "Start evaluation over entire test set" << e << " at " << time(NULL) << " s " << endl;
    EvalDataSVM(cfg, n, numBatchesTest, e*numBatches);
    cout << "Completed evaluation over entire test set" << e << " at " << time(NULL) << " s " << endl;

    network_delete(n);
    config_delete(cfg);
    return 0;
}
