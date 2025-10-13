// Fargo Project: Revised by Xi ZHAO -- Nov 16, 2022

// For PVDLB 2023: FARGO: Fast Maximum Inner Product Search via Global Multi-Probing

// For any question, please feel free to contact me. Email: xzhaoca@connect.ust.hk

#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string.h>
#include <cstring>
#include <chrono>

#include "Preprocess.h"
#include "alg.h"


extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;


int main(int argc, char const* argv[])
{
    std::string dataset = "gist";
    int varied_n = 0;
    if(argc > 1) dataset = argv[1];
    if(argc > 2) varied_n = std::atoi(argv[2]);

    std::string argvStr[4];
    argvStr[1] = (dataset);

    argvStr[3] = (dataset + ".bench_graph");

    float c = 0.9f;
    int k = 50;
    int m, L, K;

    std::cout << "Using FARGO for " << argvStr[1] << std::endl;
    Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]), varied_n);
    std::vector<resOutput> res;
    m = 1000;
    L = 5;
    K = 12;
    c = 0.3;

    int minsize_cl = 500;
    int num_cl = 10;
    int max_mst_degree = 3;

    Parameter param(prep, L, K, 1);

    lsh::timer timer;
    Partition parti(c, prep);
    // std::cout << "Partition time: " << timer.elapsed() << " s.\n" << std::endl;

    if(varied_n > 0) dataset += std::to_string(varied_n);
    argvStr[2] = (dataset + ".index");



    std::vector<int> efs = { 0,10,20,30,40,50,75,100,150,200,250,300,600,900,1200,1600,2000 };
    efs = { 100 };




    saveAndShow(c, k, dataset, res);

    return 0;
}
