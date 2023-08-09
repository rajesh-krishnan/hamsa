#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

extern "C" {
#include "hamsa.h"
}

using namespace std;

float globalTime = 0;

void EvalDataSVM(Config *cfg, Network *mynet, int numBatchesTest, int iter) {
    int Batchsize = cfg->Batchsize;

    int totCorrect = 0;
    int debugnumber = 0;
    std::ifstream testfile(cfg->testData);
    string str;
    std::getline(testfile, str); //Skip header

    ofstream outputFile(cfg->logFile,  std::ios_base::app);
    for (int i = 0; i < numBatchesTest; i++) {
        int **records = new int *[Batchsize];
        float **values = new float *[Batchsize];
        int *sizes = new int[Batchsize];
        int **labels = new int *[Batchsize];
        int *labelsize = new int[Batchsize];
        int nonzeros = 0;
        int count = 0;
        vector<string> list;
        vector<string> value;
        vector<string> label;
        while (std::getline(testfile, str)) {

            char *mystring = &str[0];
            char *pch, *pchlabel;
            int track = 0;
            list.clear();
            value.clear();
            label.clear();
            pch = strtok(mystring, " ");
            pch = strtok(NULL, " :");
            while (pch != NULL) {
                if (track % 2 == 0)
                    list.push_back(pch);
                else if (track%2==1)
                    value.push_back(pch);
                track++;
                pch = strtok(NULL, " :");
            }

            pchlabel = strtok(mystring, ",");
            while (pchlabel != NULL) {
                label.push_back(pchlabel);
                pchlabel = strtok(NULL, ",");
            }

            nonzeros += list.size();
            records[count] = new int[list.size()];
            values[count] = new float[list.size()];
            labels[count] = new int[label.size()];
            sizes[count] = list.size();
            labelsize[count] = label.size();

            int currcount = 0;
            debugnumber++;
            vector<string>::iterator it;
            for (it = list.begin(); it < list.end(); it++) {
                records[count][currcount] = stoi(*it);
                currcount++;
            }
            currcount = 0;
            for (it = value.begin(); it < value.end(); it++) {
                values[count][currcount] = stof(*it);
                currcount++;
            }
            currcount = 0;
            for (it = label.begin(); it < label.end(); it++) {
                labels[count][currcount] = stoi(*it);
                currcount++;
            }

            count++;
            if (count >= Batchsize)
                break;
        }

        int num_features = 0, num_labels = 0;
        for (int i = 0; i < Batchsize; i++)
        {
            num_features += sizes[i];
            num_labels += labelsize[i];
        }

        std::cout << Batchsize << " records, with "<< num_features << " features and " << num_labels << " labels" << std::endl;
        // auto correctPredict = 0;
        auto correctPredict = network_infer(mynet, records, values, sizes, labels, labelsize);
        totCorrect += correctPredict;
        std::cout <<" iter "<< i << ": " << totCorrect*1.0/(Batchsize*(i+1)) << " correct" << std::endl;

        delete[] sizes;
        delete[] labels;
        for (int d = 0; d < Batchsize; d++) {
            delete[] records[d];
            delete[] values[d];
        }
        delete[] records;
        delete[] values;

    }
    testfile.close();
    cout << "over all " << totCorrect * 1.0 / (numBatchesTest*Batchsize) << endl;
    outputFile << iter << " " << globalTime/1000 << " " << totCorrect * 1.0 / (numBatchesTest*Batchsize) << endl;
}

void ReadDataSVM(Config *cfg, Network* mynet, int numBatches, int epoch){
    int Batchsize = cfg->Batchsize;
    int Rebuild = cfg->Rebuild;
    int Rehash  = cfg->Rehash;
    int Stepsize  = cfg->Stepsize;

    std::ifstream file(cfg->trainData);
    std::string str;
    std::getline( file, str ); // Skip header
    for (int i = 0; i < numBatches; i++) {
        if((i+epoch*numBatches)%Stepsize==0) {
            EvalDataSVM(cfg, mynet, 20, epoch*numBatches+i);
        }
        int **records = new int *[Batchsize];
        float **values = new float *[Batchsize];
        int *sizes = new int[Batchsize];
        int **labels = new int *[Batchsize];
        int *labelsize = new int[Batchsize];
        int nonzeros = 0;
        int count = 0;
        vector<string> list;
        vector<string> value;
        vector<string> label;
        while (std::getline(file, str)) {
            char *mystring = &str[0];
            char *pch, *pchlabel;
            int track = 0;
            list.clear();
            value.clear();
            label.clear();
            pch = strtok(mystring, " ");
            pch = strtok(NULL, " :");
            while (pch != NULL) {
                if (track % 2 == 0)
                    list.push_back(pch);
                else if (track%2==1)
                    value.push_back(pch);
                track++;
                pch = strtok(NULL, " :");
            }

            pchlabel = strtok(mystring, ",");
            while (pchlabel != NULL) {
                label.push_back(pchlabel);
                pchlabel = strtok(NULL, ",");
            }

            nonzeros += list.size();
            records[count] = new int[list.size()];
            values[count] = new float[list.size()];
            labels[count] = new int[label.size()];
            sizes[count] = list.size();
            labelsize[count] = label.size();
            int currcount = 0;
            vector<string>::iterator it;
            for (it = list.begin(); it < list.end(); it++) {
                records[count][currcount] = stoi(*it);
                currcount++;
            }
            currcount = 0;
            for (it = value.begin(); it < value.end(); it++) {
                values[count][currcount] = stof(*it);
                currcount++;
            }

            currcount = 0;
            for (it = label.begin(); it < label.end(); it++) {
                labels[count][currcount] = stoi(*it);
                currcount++;
            }

            count++;
            if (count >= Batchsize)
                break;
        }

        bool rehash = false;
        bool rebuild = false;
        bool reperm = false;
        if ((epoch*numBatches+i)%(Rehash/Batchsize) == (Rehash/Batchsize-1))  rehash = true;
        if ((epoch*numBatches+i)%(Rebuild/Batchsize) == (Rehash/Batchsize-1)) rebuild = true;
        // 6496 was harcoded in Network, mvoing it here 
        if ((epoch*numBatches+i)%6946 == 6945)                                reperm = true;

        auto t1 = std::chrono::high_resolution_clock::now();
        network_train(mynet, records, values, sizes, labels, labelsize, epoch * numBatches + i,
            reperm, rehash, rebuild);
        auto t2 = std::chrono::high_resolution_clock::now();
        int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        globalTime+= timeDiffInMiliseconds;

        delete[] sizes;

        for (int d = 0; d < Batchsize; d++) {
            delete[] records[d];
            delete[] values[d];
            delete[] labels[d];
        }
        delete[] records;
        delete[] values;
        delete[] labels;

    }
    file.close();
}

int main(int argc, char* argv[])
{
    Config *cfg = config_new((char *)"sampleconfig.json");
    std::cout << "Loaded config" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    Network *n = network_new(cfg, false);
    auto t2 = std::chrono::high_resolution_clock::now();
    float timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Network Initialization takes " << timeDiffInMiliseconds/1000 << " milliseconds" << std::endl;
    int numBatches = cfg->totRecords / cfg->Batchsize;
    int numBatchesTest = cfg->totRecordsTest / cfg->Batchsize;
    int e = 0;
    while(e < cfg->Epoch) {
        ofstream outputFile(cfg->logFile, std::ios_base::app);
        outputFile<<"Epoch "<<e<<endl;
        ReadDataSVM(cfg, n, numBatches, e);
        e++;
        if(e == cfg->Epoch) {
            EvalDataSVM(cfg, n, numBatchesTest, e*numBatches);
        }else{
            EvalDataSVM(cfg, n, 50, e*numBatches);
        }
    }
    network_save_params(n);
    network_delete(n);
    config_delete(cfg);
    return 0;
}
