#pragma once

#include <chrono>

#include "ssg/index_random.h"
#include "mips/index_mips.h"
#include "ssg/util.h"

struct params{
    std::string nn_graph_path;
    unsigned L;
    unsigned R;
    unsigned sL;
    unsigned sK;
    float A;
    unsigned M;
    std::string output_path;
    std::string result_path;
};

void save_result(char* filename, std::vector<std::vector<unsigned> >& results) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);

    for(unsigned i = 0; i < results.size(); i++) {
        unsigned GK = (unsigned)results[i].size();
        out.write((char*)&GK, sizeof(unsigned));
        out.write((char*)results[i].data(), GK * sizeof(unsigned));
    }
    out.close();
}

void save_result(const char* filename, std::vector<std::vector<unsigned> >& results) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);

    for(unsigned i = 0; i < results.size(); i++) {
        unsigned GK = (unsigned)results[i].size();
        out.write((char*)&GK, sizeof(unsigned));
        out.write((char*)results[i].data(), GK * sizeof(unsigned));
    }
    out.close();
}

void read_kmeans(const std::string& filename, int query_count, std::vector<int>& query_cluster, std::vector<std::vector<int>>& init_nodes_clusters, int number_of_clusters) {
    std::ifstream file(filename, std::ios::binary);
    if(!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    query_cluster.resize(query_count);
    file.read(reinterpret_cast<char*>(query_cluster.data()), query_count * sizeof(int));

    std::vector<int> init_nodes;
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t init_nodes_count = (file_size - query_count * sizeof(int)) / sizeof(int);
    file.seekg(query_count * sizeof(int), std::ios::beg);

    init_nodes.resize(init_nodes_count);
    file.read(reinterpret_cast<char*>(init_nodes.data()), init_nodes_count * sizeof(int));

    file.close();

    size_t cluster_size = number_of_clusters;
    for(size_t i = 0; i < init_nodes.size(); i += cluster_size) {
        std::vector<int> cluster(init_nodes.begin() + i, init_nodes.begin() + i + cluster_size);
        init_nodes_clusters.push_back(cluster);
    }
}

void search(float*& data_load, float* query_load, unsigned points_num, unsigned query_num, unsigned dim, params p, Preprocess& prep) {
    // if(argc < 8) {
    //     std::cout << "./run data_file query_file ssg_path L K result_path dim"
    //         << std::endl;
    //     exit(-1);
    // }

    // std::cerr << "Data Path: " << argv[1] << std::endl;

    // unsigned points_num, dim = (unsigned)atoi(argv[7]);
    // float* data_load = nullptr;
    // data_load = efanna2e::load_data(argv[1], points_num, dim);
    // data_load = efanna2e::data_align(data_load, points_num, dim);

    // std::cerr << "Query Path: " << argv[2] << std::endl;

    // unsigned query_num, query_dim = (unsigned)atoi(argv[7]);
    // float* query_load = nullptr;
    // query_load = efanna2e::load_data(argv[2], query_num, query_dim);
    // query_load = efanna2e::data_align(query_load, query_num, query_dim);

    // assert(dim == query_dim);

    // query_load = efanna2e::data_align(query_load, query_num, dim);

    efanna2e::IndexRandom init_index(dim, points_num);
    efanna2e::IndexMips index(dim, points_num, efanna2e::FAST_L2,
        (efanna2e::Index*)(&init_index));

    std::cerr << "SSG Path: " << p.output_path << std::endl;
    std::cerr << "Result Path: " << p.result_path << std::endl;
    //std::cout << "data:" << data_load[0] << "," << data_load[1] << "," << data_load[2] << "," << data_load[3] << "," << data_load[4] << std::endl;
    index.SaveData(data_load);
    //printf("Data loaded\n");
    index.Load(p.output_path.c_str());

    unsigned L = p.sL;
    unsigned K = p.sK;

    std::cerr << "L = " << L << ", ";
    std::cerr << "K = " << K << std::endl;

    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", L);


    /* optional entry points initialization */
    // std::string filename = "../output/mnist/sn.bin";
    // std::vector<int> query_cluster;
    // std::vector<std::vector<int>> init_nodes_clusters;
    // read_kmeans(filename, query_num, query_cluster, init_nodes_clusters, 100);

    auto num = 0.0;

    std::vector<std::pair<float, float>> cal_pair;

    int Qnum = 100;
    int t = 160;

    size_t cost1 = _G_COST;
    int nq = t * Qnum;

    std::vector<std::vector<unsigned> > res(nq);
    for(unsigned i = 0; i < query_num; i++) res[i].resize(K);

    std::vector<queryN> qs;
    for(int j = 0; j < nq; j++) {
        qs.emplace_back(j % Qnum, 1, 1, prep, 1);
    }

    auto start = std::chrono::high_resolution_clock::now();


    printf("Start searching %d queries with %d threads\n", nq, 160);
#pragma omp parallel for
    for(unsigned i = 0; i < nq; i++) {
        //std::cout << "searching " << cal_inner_product(qs[i].queryPoint, data_load + 0 * dim, dim) << " th query" << std::endl;
        int dis_cal = index.Search_Mips_IP_Cal_with_No_SN(query_load + (i % Qnum) * dim, data_load, K, paras, res[i].data());
        // int dis_cal = index.Search_Mips_IP_Cal_with_No_SN(qs[i].queryPoint, data_load, K, paras, res[i].data());
        // for(auto& x : res[i]) qs[i].res.push_back(Res(x, cal_inner_product(qs[i].queryPoint, data_load + x * dim, dim)));
        num += dis_cal;
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Average Distance Computation: " << num / (int)query_num << std::endl;
    std::cout << "Average Query Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / static_cast<double>(query_num) << "ms" << std::endl;

    Performance<queryN> perform;
    for(int j = 0; j < nq; j++) {
        perform.update(qs[j], prep);
    }
    float mean_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / static_cast<double>(query_num) / nq;
    std::cout << "AVG QUERY TIME:    " << mean_time << "ms." << std::endl << std::endl;
    std::cout << "AVG RECALL:        " << ((float)perform.NN_num) / (perform.num * res[0].size()) << std::endl;
    std::cout << "AVG RATIO:         " << ((float)perform.ratio) / (perform.res_num) << std::endl;


    save_result(p.result_path.c_str(), res);

}