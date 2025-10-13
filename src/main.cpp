
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

#include "ssg/index_random.h"
#include "ssg/util.h"
#include "mips/index_mips.h"
extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;

#include "search_psp.h"
#include "rnnd.h"


void build_index(float* data_load, unsigned points_num, unsigned dim, params p){

    //std::cerr << "Data Path: " << argv[1] << std::endl;

    //unsigned points_num, dim = (unsigned)atoi(argv[8]);
    //float* data_load = nullptr;
    //data_load = efanna2e::load_data(argv[1], points_num, dim);
    data_load = efanna2e::data_align(data_load, points_num, dim);

    std::string nn_graph_path = p.nn_graph_path;
    unsigned L = p.L;
    unsigned R = p.R;
    float A = p.A;
    unsigned M = p.M;



    // unsigned L = (unsigned)atoi(argv[3]);
    // unsigned R = (unsigned)atoi(argv[4]);
    // float A = (float)atof(argv[5]);
    // unsigned M = (unsigned)atoi(argv[6]);

    std::cout << "L = " << L << ", ";
    std::cout << "R = " << R << ", ";
    std::cout << "Angle = " << A << std::endl;
    std::cout << "KNNG = " << nn_graph_path << std::endl;
    efanna2e::IndexRandom init_index(dim, points_num);
    efanna2e::IndexMips index(dim, points_num, efanna2e::L2,
        (efanna2e::Index*)(&init_index));
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<float>("A", A);
    paras.Set<unsigned>("n_try", 10);
    paras.Set<unsigned>("M", M);
    paras.Set<std::string>("nn_graph_path", nn_graph_path);

    std::cerr << "Output MIPS-SSG Path: " << p.output_path << std::endl;

    auto s = std::chrono::high_resolution_clock::now();
    index.Build(points_num, data_load, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Build Time: " << diff.count() << "\n";

    index.Save(p.output_path.c_str());

    printf("Data head1: \n");
    std::cout << data_load[0] << std::endl;
}

static inline bool exists_test(const std::string& name) {
    //return false;
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char const* argv[]){
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

    Parameter param(prep, L, K, 1);
    params p;
    p.nn_graph_path = "./indexes/" + dataset + ".knng";
    p.L = 512;
    p.R = 40;
    p.A = 60;
    p.M = 5;
    p.output_path = "./indexes/" + dataset + ".psp";
    p.sL = 150;
    p.sK = 10;
    p.result_path = "./results/" + dataset + ".txt";


    lsh::timer timer;

    if(varied_n > 0) dataset += std::to_string(varied_n);

    if(1 || !exists_test(p.nn_graph_path.c_str())){
        rnnd::rnn_para para;
        para.S = 36;
        para.T1 = 2;
        para.T2 = 4;
        std::vector<std::vector<Res>> knns;
        std::vector<std::vector<uint32_t>> knngs;
        rnnd::RNNDescent index(prep.data, para);
        index.build(prep.data.N, 1);
        index.extract_index_graph(knngs);
        index.Save(p.nn_graph_path.c_str(), knngs);
        std::cout << "knng size=" << knngs.size() << std::endl;
        std::cout << "KNNG time: " << timer.elapsed() << " s.\n" << std::endl;
    }



    if(1 || !exists_test(p.output_path.c_str())) build_index(prep.data[0], prep.data.N, prep.data.dim, p);
    std::cout << "Indexing time: " << timer.elapsed() << " s.\n" << std::endl;

    // printf("Data head2: \n");
    // std::cout << prep.data[1][2] << std::endl;

    // std::cout << prep.data[0][0] << std::endl;
    search(prep.data[0], prep.queries[0], prep.data.N, 100, prep.data.dim, p, prep);

    std::cout << "Searching time: " << timer.elapsed() << " s.\n" << std::endl;
    std::vector<int> efs = { 0,10,20,30,40,50,75,100,150,200,250,300,600,900,1200,1600,2000 };
    efs = { 100 };

    saveAndShow(c, k, dataset, res);

    return 0;
}
