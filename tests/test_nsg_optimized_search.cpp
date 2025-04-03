//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>
#include <omp.h>
#include <iomanip>

void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  // std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

// Sungjun Jung: Read groundtruth 
void load_data_ivecs(char* filename, unsigned int*& data, unsigned& num,
               unsigned& dim) { 
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new unsigned int[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

void save_result(const char* filename,
                 std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}
int main(int argc, char** argv) {
  if (argc != 11) {
    std::cout << argv[0]
              << " data_file query_file groundtruth_file nsg_path search_L search_K result_path num_threads tau hash_bitwidth"
              << std::endl;
    exit(-1);
  }
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  assert(dim == query_dim);

  unsigned L = (unsigned)atoi(argv[5]);
  unsigned K = (unsigned)atoi(argv[6]);

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  // Sungjun Jung: Load groundtruth to compute recall
  uint32_t* ground_truth_load = NULL;
  uint32_t ground_truth_num, ground_truth_dim;
  load_data_ivecs(argv[3], ground_truth_load, ground_truth_num, ground_truth_dim);

  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build query_load = efanna2e::data_align(query_load,
  // query_num, query_dim);
  efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
  index.Load(argv[4]);

  // Sungjun Jung: Setting parameters for ADA-NNS
  float tau = (float)atof(argv[9]);
  uint64_t hash_bitwidth = (uint64_t)atoi(argv[10]);
  index.SetHashBitwidth(hash_bitwidth);
  index.SetTau(tau);

  index.OptimizeGraph(data_load);

  // Sungjun Jung: Generate/Load data for ADA-NNS
  char* hash_function_name = new char[strlen(argv[1]) + strlen(".hash_function_") + strlen(argv[10]) + 1];
  char* hashed_set_name = new char[strlen(argv[1]) + strlen(".hashed_set") + strlen(argv[10]) + 1];
  strcpy(hash_function_name, argv[1]);
  strcat(hash_function_name, ".hash_function_");
  strcat(hash_function_name, argv[10]);
  strcat(hash_function_name, "b");
  strcpy(hashed_set_name, argv[1]);
  strcat(hashed_set_name, ".hashed_set_");
  strcat(hashed_set_name, argv[10]);
  strcat(hashed_set_name, "b");
  if (index.ReadHashFunction(hash_function_name)) {
    if (!index.ReadHashedSet(hashed_set_name))
      index.GenerateHashedSet(hashed_set_name, data_load);
  }
  else {
    index.GenerateHashFunction(hash_function_name);
    index.GenerateHashedSet(hashed_set_name, data_load);
  }
  delete[] hash_function_name;
  delete[] hashed_set_name;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);

  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

  uint32_t num_threads = (uint32_t)atoi(argv[8]);
  omp_set_num_threads(num_threads);

  auto s = std::chrono::high_resolution_clock::now();
  // Sungjun Jung: Hash query vector
  uint32_t* hashed_query_buffer = (uint32_t*)malloc(query_num * (hash_bitwidth >> 3));
  index.QueryHash(query_load, hashed_query_buffer, query_num);
#pragma omp parallel for schedule(dynamic, 1)
  for (unsigned i = 0; i < query_num; i++) {
    index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data(), hashed_query_buffer + (hash_bitwidth >> 5) * i);
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
//  std::cout << "search time: " << diff.count() << "\n";

  save_result(argv[7], res);

  // Sungjun Jung: Evaluate Recall
  unsigned int topk_hit = 0;
  for (unsigned int i = 0; i < query_num; i++) {
    for (unsigned int j = 0; j < K; j++) {
      for (unsigned int k = 0; k < K; k++) {
        if (res[i][j] == *(ground_truth_load + i * ground_truth_dim + k)) {
          topk_hit++;
          break;
        }
      }
    }
  }
  float recall = (float)topk_hit / (query_num * K) * 100;

  std::cout << std::left << std::setw(5) << L << std::setw(10) << query_num / diff.count() << std::setw(10) << recall << std::endl;

  return 0;
}
