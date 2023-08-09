#include <chrono>
#include <iostream>

extern "C" {
#include "hamsa.h"
}

using namespace std;

#if 0
void EvalDataSVM(int numBatchesTest,  Network* _mynet, int iter){
    int totCorrect = 0;
    int debugnumber = 0;
    std::ifstream testfile(testData);
    string str;
    //Skipe header
    std::getline( testfile, str );

    ofstream outputFile(logFile,  std::ios_base::app);
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
        auto correctPredict = _mynet->predictClass(records, values, sizes, labels, labelsize);
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

void ReadDataSVM(size_t numBatches,  Network* _mynet, int epoch){
    std::ifstream file(trainData);
    std::string str;
    //skipe header
    std::getline( file, str );
    for (size_t i = 0; i < numBatches; i++) {
        if((i+epoch*numBatches)%Stepsize==0) {
            EvalDataSVM(20, _mynet, epoch*numBatches+i);
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
        if ((epoch*numBatches+i)%(Rehash/Batchsize) == ((size_t)Rehash/Batchsize-1)){
            if(Mode==1 || Mode==4) {
                rehash = true;
            }
        }

        if ((epoch*numBatches+i)%(Rebuild/Batchsize) == ((size_t)Rehash/Batchsize-1)){
            if(Mode==1 || Mode==4) {
                rebuild = true;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        // logloss
        _mynet->ProcessInput(records, values, sizes, labels, labelsize, epoch * numBatches + i,
                                            rehash, rebuild);

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
#endif

int main(int argc, char* argv[])
{
    //parseconfig(argv[1]);

    // int numBatches = totRecords/Batchsize;
    // int numBatchesTest = totRecordsTest/Batchsize;

    auto t1 = std::chrono::high_resolution_clock::now();
    Config *cfg = config_new((char *)"sampleconfig.json");
    Network *n = network_new(cfg, false);
    auto t2 = std::chrono::high_resolution_clock::now();
    float timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Network Initialization takes " << timeDiffInMiliseconds/1000 << " milliseconds" << std::endl;

#if 0
    //***********************************
    // Start Training
    //***********************************

    for (int e=0; e< Epoch; e++) {
        ofstream outputFile(logFile,  std::ios_base::app);
        outputFile<<"Epoch "<<e<<endl;
        // train
        ReadDataSVM(numBatches, _mynet, e);

        // test
        if(e==Epoch-1) {
            EvalDataSVM(numBatchesTest, _mynet, (e+1)*numBatches);
        }else{
            EvalDataSVM(50, _mynet, (e+1)*numBatches);
        }
        _mynet->saveWeights(savedWeights);

    }

    delete [] RangePow;
    delete [] K;
    delete [] L;
    delete [] Sparsity;
#endif

    return 0;
}
