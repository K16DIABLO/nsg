#include "efanna2e/index_nsg.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>

#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"

namespace efanna2e {
#define _CONTROL_NUM 100
IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m,
                   Index *initializer)
    : Index(dimension, n, m), initializer_{initializer} {}

IndexNSG::~IndexNSG() {
    if (distance_ != nullptr) {
        delete distance_;
        distance_ = nullptr;
    }
    if (initializer_ != nullptr) {
        delete initializer_;
        initializer_ = nullptr;
    }
    if (opt_graph_ != nullptr) {
        delete opt_graph_;
        opt_graph_ = nullptr;
    }
}

void IndexNSG::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *)&width, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    unsigned GK = (unsigned)final_graph_[i].size();
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void IndexNSG::Load(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&width, sizeof(unsigned));
  in.read((char *)&ep_, sizeof(unsigned));
  // width=100;
  unsigned cc = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof()) break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *)tmp.data(), k * sizeof(unsigned));
    final_graph_.push_back(tmp);
  }
  cc /= nd_;
  // std::cout<<cc<<std::endl;
}
void IndexNSG::Load_nn_graph(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  in.read((char *)&k, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t num = (unsigned)(fsize / (k + 1) / 4);
  in.seekg(0, std::ios::beg);

  final_graph_.resize(num);
  final_graph_.reserve(num);
  unsigned kk = (k + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(k);
    final_graph_[i].reserve(kk);
    in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
  }
  in.close();
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  boost::dynamic_bitset<> flags{nd_, 0};
  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             boost::dynamic_bitset<> &flags,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    fullset.push_back(retset[i]);
    // flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::init_graph(const Parameters &parameters) {
  float *center = new float[dimension_];
  for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
  for (unsigned i = 0; i < nd_; i++) {
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] += data_[i * dimension_ + j];
    }
  }
  for (unsigned j = 0; j < dimension_; j++) {
    center[j] /= nd_;
  }
  std::vector<Neighbor> tmp, pool;
  ep_ = rand() % nd_;  // random initialize navigating point
  get_neighbors(center, parameters, tmp, pool);
  ep_ = tmp[0].id;
  delete center;
}

void IndexNSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                          const Parameters &parameter,
                          boost::dynamic_bitset<> &flags,
                          SimpleNeighbor *cut_graph_) {
  unsigned range = parameter.Get<unsigned>("R");
  unsigned maxc = parameter.Get<unsigned>("C");
  width = range;
  unsigned start = 0;

  for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
    unsigned id = final_graph_[q][nn];
    if (flags[id]) continue;
    float dist =
        distance_->compare(data_ + dimension_ * (size_t)q,
                           data_ + dimension_ * (size_t)id, (unsigned)dimension_);
    pool.push_back(Neighbor(id, dist, true));
  }

  std::sort(pool.begin(), pool.end());
  std::vector<Neighbor> result;
  if (pool[start].id == q) start++;
  result.push_back(pool[start]);

  while (result.size() < range && (++start) < pool.size() && start < maxc) {
    auto &p = pool[start];
    bool occlude = false;
    for (unsigned t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                     data_ + dimension_ * (size_t)p.id,
                                     (unsigned)dimension_);
      if (djk < p.distance /* dik */) {
        occlude = true;
        break;
      }
    }
    if (!occlude) result.push_back(p);
  }

  SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
  for (size_t t = 0; t < result.size(); t++) {
    des_pool[t].id = result[t].id;
    des_pool[t].distance = result[t].distance;
  }
  if (result.size() < range) {
    des_pool[result.size()].distance = -1;
  }
}

void IndexNSG::InterInsert(unsigned n, unsigned range,
                           std::vector<std::mutex> &locks,
                           SimpleNeighbor *cut_graph_) {
  SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
  for (size_t i = 0; i < range; i++) {
    if (src_pool[i].distance == -1) break;

    SimpleNeighbor sn(n, src_pool[i].distance);
    size_t des = src_pool[i].id;
    SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

    std::vector<SimpleNeighbor> temp_pool;
    int dup = 0;
    {
      LockGuard guard(locks[des]);
      for (size_t j = 0; j < range; j++) {
        if (des_pool[j].distance == -1) break;
        if (n == des_pool[j].id) {
          dup = 1;
          break;
        }
        temp_pool.push_back(des_pool[j]);
      }
    }
    if (dup) continue;

    temp_pool.push_back(sn);
    if (temp_pool.size() > range) {
      std::vector<SimpleNeighbor> result;
      unsigned start = 0;
      std::sort(temp_pool.begin(), temp_pool.end());
      result.push_back(temp_pool[start]);
      while (result.size() < range && (++start) < temp_pool.size()) {
        auto &p = temp_pool[start];
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
          float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                         data_ + dimension_ * (size_t)p.id,
                                         (unsigned)dimension_);
          if (djk < p.distance /* dik */) {
            occlude = true;
            break;
          }
        }
        if (!occlude) result.push_back(p);
      }
      {
        LockGuard guard(locks[des]);
        for (unsigned t = 0; t < result.size(); t++) {
          des_pool[t] = result[t];
        }
      }
    } else {
      LockGuard guard(locks[des]);
      for (unsigned t = 0; t < range; t++) {
        if (des_pool[t].distance == -1) {
          des_pool[t] = sn;
          if (t + 1 < range) des_pool[t + 1].distance = -1;
          break;
        }
      }
    }
  }
}

void IndexNSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
  /*
  std::cout << " graph link" << std::endl;
  unsigned progress=0;
  unsigned percent = 100;
  unsigned step_size = nd_/percent;
  std::mutex progress_lock;
  */
  unsigned range = parameters.Get<unsigned>("R");
  std::vector<std::mutex> locks(nd_);

#pragma omp parallel
  {
    // unsigned cnt = 0;
    std::vector<Neighbor> pool, tmp;
    boost::dynamic_bitset<> flags{nd_, 0};
#pragma omp for schedule(dynamic, 100)
    for (unsigned n = 0; n < nd_; ++n) {
      pool.clear();
      tmp.clear();
      flags.reset();
      get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
      sync_prune(n, pool, parameters, flags, cut_graph_);
      /*
    cnt++;
    if(cnt % step_size == 0){
      LockGuard g(progress_lock);
      std::cout<<progress++ <<"/"<< percent << " completed" << std::endl;
      }
      */
    }
  }

#pragma omp for schedule(dynamic, 100)
  for (unsigned n = 0; n < nd_; ++n) {
    InterInsert(n, range, locks, cut_graph_);
  }
}

void IndexNSG::Build(size_t n, const float *data, const Parameters &parameters) {
  std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
  unsigned range = parameters.Get<unsigned>("R");
  Load_nn_graph(nn_graph_path.c_str());
  data_ = data;
  init_graph(parameters);
  SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
  Link(parameters, cut_graph_);
  final_graph_.resize(nd_);

  for (size_t i = 0; i < nd_; i++) {
    SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
    unsigned pool_size = 0;
    for (unsigned j = 0; j < range; j++) {
      if (pool[j].distance == -1) break;
      pool_size = j;
    }
    pool_size++;
    final_graph_[i].resize(pool_size);
    for (unsigned j = 0; j < pool_size; j++) {
      final_graph_[i][j] = pool[j].id;
    }
  }

  tree_grow(parameters);

  unsigned max = 0, min = 1e6, avg = 0;
  for (size_t i = 0; i < nd_; i++) {
    auto size = final_graph_[i].size();
    max = max < size ? size : max;
    min = min > size ? size : min;
    avg += size;
  }
  avg /= 1.0 * nd_;
  printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);

  has_built = true;
  delete cut_graph_;
}

void IndexNSG::Search(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  data_ = x;
  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  unsigned tmp_l = 0;
  for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
    init_ids[tmp_l] = final_graph_[ep_][tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    float dist =
        distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = true;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;
        float dist =
            distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexNSG::SearchWithOptGraph(const float *query, size_t K,
                                  const Parameters &parameters, unsigned *indices,
                                  uint32_t* hashed_query_buffer = nullptr) {
  unsigned L = parameters.Get<unsigned>("L_search");
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;

  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned tmp_l = 0;
  unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * ep_ + data_len);
  unsigned MaxM_ep = *neighbors;
  neighbors++;

  // Sungjun Jung: Copy hashed query
  float query_norm = sqrt(dist_fast->norm(query, (unsigned)dimension_));
  std::vector<SimpleNeighbor> selected_pool(width);
  uint64_t hash_size = hash_bitwidth_ >> 5;
  uint32_t* hashed_query = new uint32_t[hash_size];
  memcpy(hashed_query, hashed_query_buffer, (this->hash_bitwidth_ >> 3));

  for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
    init_ids[tmp_l] = neighbors[tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
  }
  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float *x = (float *)(opt_graph_ + node_size * id);
    float norm_x = *x;
    x++;
    float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    flags[id] = true;
    L++;
  }
  // std::cout<<L<<std::endl;

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);

      // Sungjun Jun: Candidate selection for ADA-NNS
      uint32_t selected_pool_size = CandidateSelection(query_norm, hashed_query, selected_pool, flags, neighbors);
      for (unsigned m = 0; m < selected_pool_size; ++m)
        _mm_prefetch(opt_graph_ + node_size * selected_pool[m].id, _MM_HINT_T0);
      for (unsigned m = 0; m < selected_pool_size; ++m) {
        unsigned id = selected_pool[m].id;
        if (flags[id]) continue;
        flags[id] = 1;
        float *data = (float *)(opt_graph_ + node_size * id);
        float norm = *data;
        data++;
        float dist = dist_fast->compare(query, data, norm, (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        // if(L+1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
  delete[] hashed_query;
}

void IndexNSG::OptimizeGraph(float *data) {  // use after build or load

  data_ = data;
  data_len = (dimension_ + 1) * sizeof(float);
  neighbor_len = (width + 1) * sizeof(unsigned);
  node_size = data_len + neighbor_len;

  // Sungjun Jung: Memory size for hash used by ADA-NNS
  uint64_t hash_len = (hash_bitwidth_ >> 3) + 2 * sizeof(float);
  uint64_t hash_function_size = dimension_ * hash_bitwidth_ * sizeof(float);
  uint64_t cosine_table_size = hash_bitwidth_ * sizeof(float);
  opt_graph_ = (char *)malloc(node_size * nd_ + hash_len * nd_ + hash_function_size + cosine_table_size);
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
  for (unsigned i = 0; i < nd_; i++) {
    char *cur_node_offset = opt_graph_ + i * node_size;
    float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
    std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
    std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                data_len - sizeof(float));

    cur_node_offset += data_len;
    unsigned k = final_graph_[i].size();
    std::memcpy(cur_node_offset, &k, sizeof(unsigned));
    std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
                k * sizeof(unsigned));
    std::vector<unsigned>().swap(final_graph_[i]);

    // Sungjun Jung: Copy norm and sqrt_norm of base set
    cur_node_offset = opt_graph_ + nd_ * node_size + i * hash_len;
    float sqrt_cur_norm = sqrt(cur_norm);
    std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
    std::memcpy(cur_node_offset + sizeof(float), &sqrt_cur_norm, sizeof(float));
  }

  // Sungjun Jung: Generate cosine table for ADA-NNS
  for (unsigned i = 0; i < hash_bitwidth_; i++) {
    char* cur_offset = opt_graph_ + (node_size + hash_len) * nd_ + hash_function_size + i * sizeof(float);
    float cosine_value = cos(i * M_PI / hash_bitwidth_);
    std::memcpy(cur_offset, &cosine_value, sizeof(float));
  }
  CompactGraph().swap(final_graph_);
}

void IndexNSG::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
  unsigned tmp = root;
  std::stack<unsigned> s;
  s.push(root);
  if (!flag[root]) cnt++;
  flag[root] = true;
  while (!s.empty()) {
    unsigned next = nd_ + 1;
    for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
      if (flag[final_graph_[tmp][i]] == false) {
        next = final_graph_[tmp][i];
        break;
      }
    }
    // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
    if (next == (nd_ + 1)) {
      s.pop();
      if (s.empty()) break;
      tmp = s.top();
      continue;
    }
    tmp = next;
    flag[tmp] = true;
    s.push(tmp);
    cnt++;
  }
}

void IndexNSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                        const Parameters &parameter) {
  unsigned id = nd_;
  for (unsigned i = 0; i < nd_; i++) {
    if (flag[i] == false) {
      id = i;
      break;
    }
  }

  if (id == nd_) return;  // No Unlinked Node

  std::vector<Neighbor> tmp, pool;
  get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
  std::sort(pool.begin(), pool.end());

  unsigned found = 0;
  for (unsigned i = 0; i < pool.size(); i++) {
    if (flag[pool[i].id]) {
      // std::cout << pool[i].id << '\n';
      root = pool[i].id;
      found = 1;
      break;
    }
  }
  if (found == 0) {
    while (true) {
      unsigned rid = rand() % nd_;
      if (flag[rid]) {
        root = rid;
        break;
      }
    }
  }
  final_graph_[root].push_back(id);
}
void IndexNSG::tree_grow(const Parameters &parameter) {
  unsigned root = ep_;
  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned unlinked_cnt = 0;
  while (unlinked_cnt < nd_) {
    DFS(flags, root, unlinked_cnt);
    // std::cout << unlinked_cnt << '\n';
    if (unlinked_cnt >= nd_) break;
    findroot(flags, root, parameter);
    // std::cout << "new root"<<":"<<root << '\n';
  }
  for (size_t i = 0; i < nd_; ++i) {
    if (final_graph_[i].size() > width) {
      width = final_graph_[i].size();
    }
  }
}

// Sungjun Jung: Below are ADA-NNS functions
void IndexNSG::GenerateHashFunction (char* file_name) {
  DistanceFastL2* dist_fast = (DistanceFastL2*) distance_;
  std::normal_distribution<float> norm_dist (0.0, 1.0);
  std::mt19937 gen(rand());
  uint64_t hash_len = (hash_bitwidth_ >> 3) + 2 * sizeof(float);
  hash_function_ = (float*)(opt_graph_ + node_size * nd_ + hash_len * nd_);
  float hash_function_norm;
  uint32_t num_superbit_blocks = (hash_bitwidth_ >> 6);
  uint32_t superbit_batch_size = hash_bitwidth_ / num_superbit_blocks;

//  std::cout << "GenerateHashFunction" << std::endl;
//  auto s = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < hash_bitwidth_; i+=superbit_batch_size) {
    // Gram-schmidt process
    for (uint32_t dim = 0; dim < dimension_; dim++) {
      hash_function_[i * dimension_ + dim] = norm_dist(gen);
    }
    hash_function_norm = std::sqrt(dist_fast->norm(&hash_function_[i * dimension_], dimension_));
    for (uint32_t dim = 0; dim < dimension_; dim++) {
      hash_function_[i * dimension_ + dim] /= hash_function_norm;
    }

    for (uint32_t hash_col = 1; hash_col < superbit_batch_size; hash_col++) { 
      for (unsigned int dim = 0; dim < dimension_; dim++) {
        hash_function_[(i + hash_col) * dimension_ + dim] = norm_dist(gen);
      }

      // Gram-schmidt process
      for (uint32_t compare_col = 0; compare_col < hash_col; compare_col++) {
        float inner_product_between_hash = dist_fast->DistanceInnerProduct::compare(&hash_function_[(i + hash_col) * dimension_], &hash_function_[(i + compare_col) * dimension_], (uint32_t)dimension_);
        for (uint32_t dim = 0; dim < dimension_; dim++) {
          hash_function_[(i + hash_col) * dimension_ + dim] -= (inner_product_between_hash * hash_function_[(i + compare_col) * dimension_ + dim]);
        }
      }
      hash_function_norm = std::sqrt(dist_fast->norm(&hash_function_[(i + hash_col) * dimension_], dimension_));
      for (uint32_t dim = 0; dim < dimension_; dim++) {
        hash_function_[(i + hash_col) * dimension_ + dim] /= hash_function_norm;
      }
    }
  }
//  auto e = std::chrono::high_resolution_clock::now();
//  std::chrono::duration<double> diff = e - s;
//    std::cout << "HashFunction generation time: " << diff.count() * 1000 << std::endl;;

  std::ofstream file_hash_function(file_name, std::ios::binary | std::ios::out);
  file_hash_function.write((char*)&hash_bitwidth_, sizeof(uint32_t));
  file_hash_function.write((char*)hash_function_, dimension_ * hash_bitwidth_ * sizeof(float));
  file_hash_function.close();
}
void IndexNSG::GenerateHashedSet (char* file_name, float* data) {
  uint64_t hash_len = (hash_bitwidth_ >> 3) + 2 * sizeof(float);

//  std::cout << "GenerateHashedSet" << std::endl;
//  auto s = std::chrono::high_resolution_clock::now();
  Eigen::setNbThreads(omp_get_num_procs());
  Eigen::Map<Eigen::MatrixXf> eigen_mat0(data, this->dimension_, this->nd_);
  Eigen::Map<Eigen::MatrixXf> eigen_mat1(this->hash_function_, this->dimension_, this->hash_bitwidth_);

  Eigen::MatrixXf result = eigen_mat1.transpose() * eigen_mat0;

  for (size_t i = 0; i < this->nd_; i++) {
    for (size_t j = 0; j < (this->hash_bitwidth_ >> 5); j++) {
      uint32_t bin_value = 0;

      bin_value |= ((result(j * 32 + 0, i) > 0) << 0);
      bin_value |= ((result(j * 32 + 1, i) > 0) << 1);
      bin_value |= ((result(j * 32 + 2, i) > 0) << 2);
      bin_value |= ((result(j * 32 + 3, i) > 0) << 3);
      bin_value |= ((result(j * 32 + 4, i) > 0) << 4);
      bin_value |= ((result(j * 32 + 5, i) > 0) << 5);
      bin_value |= ((result(j * 32 + 6, i) > 0) << 6);
      bin_value |= ((result(j * 32 + 7, i) > 0) << 7);
      bin_value |= ((result(j * 32 + 8, i) > 0) << 8);
      bin_value |= ((result(j * 32 + 9, i) > 0) << 9);
      bin_value |= ((result(j * 32 + 10, i) > 0) << 10);
      bin_value |= ((result(j * 32 + 11, i) > 0) << 11);
      bin_value |= ((result(j * 32 + 12, i) > 0) << 12);
      bin_value |= ((result(j * 32 + 13, i) > 0) << 13);
      bin_value |= ((result(j * 32 + 14, i) > 0) << 14);
      bin_value |= ((result(j * 32 + 15, i) > 0) << 15);
      bin_value |= ((result(j * 32 + 16, i) > 0) << 16);
      bin_value |= ((result(j * 32 + 17, i) > 0) << 17);
      bin_value |= ((result(j * 32 + 18, i) > 0) << 18);
      bin_value |= ((result(j * 32 + 19, i) > 0) << 19);
      bin_value |= ((result(j * 32 + 20, i) > 0) << 20);
      bin_value |= ((result(j * 32 + 21, i) > 0) << 21);
      bin_value |= ((result(j * 32 + 22, i) > 0) << 22);
      bin_value |= ((result(j * 32 + 23, i) > 0) << 23);
      bin_value |= ((result(j * 32 + 24, i) > 0) << 24);
      bin_value |= ((result(j * 32 + 25, i) > 0) << 25);
      bin_value |= ((result(j * 32 + 26, i) > 0) << 26);
      bin_value |= ((result(j * 32 + 27, i) > 0) << 27);
      bin_value |= ((result(j * 32 + 28, i) > 0) << 28);
      bin_value |= ((result(j * 32 + 29, i) > 0) << 29);
      bin_value |= ((result(j * 32 + 30, i) > 0) << 30);
      bin_value |= ((result(j * 32 + 31, i) > 0) << 31);

      this->hashed_set_ = (uint32_t*)(opt_graph_ + node_size * nd_ + hash_len * i + 2 * sizeof(float));
      *(this->hashed_set_ + j) = bin_value;
    }
  }
//  auto e = std::chrono::high_resolution_clock::now();
//  std::chrono::duration<double> diff = e - s;
//    std::cout << "HashedSet generation time: " << diff.count() * 1000 << std::endl;;

  std::ofstream file_hashed_set(file_name, std::ios::binary | std::ios::out);
  hashed_set_ = (uint32_t*)(opt_graph_ + node_size * nd_);
  for (size_t i = 0; i < nd_; i++) {
    file_hashed_set.write((char*)(opt_graph_ + node_size * nd_ + hash_len * i + 2 * sizeof(float)), hash_len - 2 * sizeof(float));
  }
  file_hashed_set.close();
}
bool IndexNSG::ReadHashFunction (char* file_name) {
  std::ifstream file_hash_function(file_name, std::ios::binary);
  uint64_t hash_len = (hash_bitwidth_ >> 3) + 2 * sizeof(float);
  if (file_hash_function.is_open()) {
    uint32_t hash_bitwidth_temp;
    file_hash_function.read((char*)&hash_bitwidth_temp, sizeof(uint32_t));
    if (hash_bitwidth_ != hash_bitwidth_temp) {
      file_hash_function.close();
      return false;
    }

//    std::cout << "ReadHashFunction" << std::endl;
    hash_function_ = (float*)(opt_graph_ + node_size * nd_ + hash_len * nd_);
    file_hash_function.read((char*)hash_function_, dimension_ * hash_bitwidth_ * sizeof(float));
    file_hash_function.close();
    return true;
  }
  else
    return false;
}
bool IndexNSG::ReadHashedSet (char* file_name) {
  std::ifstream file_hashed_set(file_name, std::ios::binary);
  uint64_t hash_len = (hash_bitwidth_ >> 3) + 2 * sizeof(float);
  if (file_hashed_set.is_open()) {
//    std::cout << "ReadHashedSet" << std::endl;
    hashed_set_ = (uint32_t*)(opt_graph_ + node_size * nd_);
    for (uint32_t i = 0; i < nd_; i++) {
      file_hashed_set.read((char*)(opt_graph_ + node_size * nd_ + hash_len * i + 2 * sizeof(float)), hash_len - 2 * sizeof(float));
    }
    file_hashed_set.close();
    
    return true;
  }
  else
    return false;
}
void IndexNSG::QueryHash (const float* query, uint32_t* hashed_query, const uint64_t num_query) {
  Eigen::setNbThreads(omp_get_num_threads());
  Eigen::Map<const Eigen::MatrixXf> eigen_mat0(query, this->dimension_, num_query);
  Eigen::Map<const Eigen::MatrixXf> eigen_mat1(this->hash_function_, this->dimension_, this->hash_bitwidth_);

  Eigen::MatrixXf result = eigen_mat1.transpose() * eigen_mat0;

  for (size_t i = 0; i < num_query; i++) {
    for (size_t j = 0; j < (this->hash_bitwidth_ >> 5); j++) {
      uint32_t bin_value = 0;

      bin_value |= ((result(j * 32 + 0, i) > 0) << 0);
      bin_value |= ((result(j * 32 + 1, i) > 0) << 1);
      bin_value |= ((result(j * 32 + 2, i) > 0) << 2);
      bin_value |= ((result(j * 32 + 3, i) > 0) << 3);
      bin_value |= ((result(j * 32 + 4, i) > 0) << 4);
      bin_value |= ((result(j * 32 + 5, i) > 0) << 5);
      bin_value |= ((result(j * 32 + 6, i) > 0) << 6);
      bin_value |= ((result(j * 32 + 7, i) > 0) << 7);
      bin_value |= ((result(j * 32 + 8, i) > 0) << 8);
      bin_value |= ((result(j * 32 + 9, i) > 0) << 9);
      bin_value |= ((result(j * 32 + 10, i) > 0) << 10);
      bin_value |= ((result(j * 32 + 11, i) > 0) << 11);
      bin_value |= ((result(j * 32 + 12, i) > 0) << 12);
      bin_value |= ((result(j * 32 + 13, i) > 0) << 13);
      bin_value |= ((result(j * 32 + 14, i) > 0) << 14);
      bin_value |= ((result(j * 32 + 15, i) > 0) << 15);
      bin_value |= ((result(j * 32 + 16, i) > 0) << 16);
      bin_value |= ((result(j * 32 + 17, i) > 0) << 17);
      bin_value |= ((result(j * 32 + 18, i) > 0) << 18);
      bin_value |= ((result(j * 32 + 19, i) > 0) << 19);
      bin_value |= ((result(j * 32 + 20, i) > 0) << 20);
      bin_value |= ((result(j * 32 + 21, i) > 0) << 21);
      bin_value |= ((result(j * 32 + 22, i) > 0) << 22);
      bin_value |= ((result(j * 32 + 23, i) > 0) << 23);
      bin_value |= ((result(j * 32 + 24, i) > 0) << 24);
      bin_value |= ((result(j * 32 + 25, i) > 0) << 25);
      bin_value |= ((result(j * 32 + 26, i) > 0) << 26);
      bin_value |= ((result(j * 32 + 27, i) > 0) << 27);
      bin_value |= ((result(j * 32 + 28, i) > 0) << 28);
      bin_value |= ((result(j * 32 + 29, i) > 0) << 29);
      bin_value |= ((result(j * 32 + 30, i) > 0) << 30);
      bin_value |= ((result(j * 32 + 31, i) > 0) << 31);

      hashed_query[(this->hash_bitwidth_ >> 5) * i + j] = bin_value;
    }
  }
}
uint32_t IndexNSG::CandidateSelection (const float query_norm, const uint32_t* hashed_query, std::vector<SimpleNeighbor>& selected_pool, boost::dynamic_bitset<>& flags, const uint32_t* neighbors) {
  uint32_t MaxM = *neighbors;
  neighbors++;
  uint32_t new_MaxM = 0;
  uint32_t selected_pool_size_limit = (uint32_t)ceil(MaxM * tau_);
  uint64_t hash_len = (hash_bitwidth_ >> 3) + 2 * sizeof(float);
  uint64_t hash_function_size = dimension_ * hash_bitwidth_ * sizeof(float);
  float* cosine_table = (float*)(opt_graph_ + (node_size + hash_len) * nd_ + hash_function_size); 

  uint32_t filter_visited = 0;
  std::vector<SimpleNeighbor> filter_pool;

  for (uint32_t m = 0; m < MaxM; m += 16) {
    _mm_prefetch(neighbors + m, _MM_HINT_T0);
  }

  for (uint32_t m = 0; m < MaxM; ++m) {
    uint32_t id = neighbors[m];
    if (flags[id]) {
      filter_visited++;
      continue;
    }
    selected_pool[new_MaxM].id = id;
    for (uint64_t n = 0; n < hash_len; n += 64)
      _mm_prefetch(opt_graph_ + node_size * nd_ + hash_len * id + n, _MM_HINT_T0);
    new_MaxM++;
  }
  if (new_MaxM < selected_pool_size_limit) return new_MaxM;

  uint64_t hamming_result[4];
  uint32_t selected_pool_size = 0;
  std::vector<SimpleNeighbor>::iterator index;

  for (uint32_t m = 0; m < new_MaxM; ++m) {
    uint32_t id = selected_pool[m].id;
    uint32_t hamming_distance = 0;
    uint32_t* hashed_set_address = (uint32_t*)(opt_graph_ + node_size * nd_ + hash_len * id);
    float norm = *(float*)hashed_set_address;
    hashed_set_address++;
    float sqrt_norm = *(float*)hashed_set_address;
    hashed_set_address++;
    float mul_query_base = sqrt_norm * query_norm;
#ifdef __AVX__
    for (uint32_t i = 0; i < (hash_bitwidth_ >> 8); i++) {
      // XOR
      __m256i hashed_query_avx;
      __m256i hashed_set_avx;
      __m256i hamming_result_avx;
      hashed_query_avx = _mm256_loadu_si256((__m256i*)&hashed_query[i << 3]);
      hashed_set_avx = _mm256_loadu_si256((__m256i*)(hashed_set_address));
      hamming_result_avx = _mm256_xor_si256(hashed_query_avx, hashed_set_avx);
      // Count 1s
      _mm256_storeu_si256((__m256i*)&hamming_result, hamming_result_avx);
      hamming_distance += _mm_popcnt_u64(hamming_result[0]);
      hamming_distance += _mm_popcnt_u64(hamming_result[1]);
      hamming_distance += _mm_popcnt_u64(hamming_result[2]);
      hamming_distance += _mm_popcnt_u64(hamming_result[3]);
      hashed_set_address += 8;
    }
#else
    for (uint32_t num_integer = 0; num_integer < hash_bitwidth_ / (8 * sizeof(uint32_t)); num_integer++) {
      uint32_t* hashed_set = (uint32_t*)(opt_graph_ + node_size * id + data_len + neighbor_len);
      hamming_result[num_integer] = hashed_query[num_integer] ^ hashed_set[num_integer]; 
      hamming_distance += __builtin_popcount(hamming_result[num_integer]);
    }
#endif
    float distance = - norm + 2 * mul_query_base * cosine_table[hamming_distance];
    SimpleNeighbor cat_hamming_id(id, distance);
    if ((selected_pool_size_limit == selected_pool_size) && (distance > index->distance)) {
      *index = cat_hamming_id;
      index = std::min_element(selected_pool.begin(), selected_pool.begin() + selected_pool_size_limit);
    }

    if (selected_pool_size < selected_pool_size_limit) {
      selected_pool[selected_pool_size] = cat_hamming_id;
      selected_pool_size++;
      if (selected_pool_size == selected_pool_size_limit) {
        index = std::min_element(selected_pool.begin(), selected_pool.begin() + selected_pool_size_limit);
      }
    }
  }
  return selected_pool_size_limit;
}
}
