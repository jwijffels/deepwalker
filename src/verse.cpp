#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <Rcpp.h>
#ifdef _WIN32
#define _MSC_VER 1 
#endif

using namespace std;

#if defined(__AVX2__) || \
defined(__FMA__)
#define VECTORIZE 1
#define AVX_LOOP _Pragma("omp simd")
#else
#define AVX_LOOP
#endif

#ifndef UINT64_C
#define UINT64_C(c) (c##ULL)
#endif

#define SIGMOID_BOUND 6.0
#define DEFAULT_ALIGN 128

typedef unsigned long long ull;

bool silent = false;
int n_threads = 1;
float global_lr = 0.0025f;
int n_epochs = 100000;
int n_hidden = 128;
int n_samples = 3;
float ppralpha = 0.85f;

ull total_steps;
ull step = 0;

ull nv = 0, ne = 0;
int *offsets;
int *edges;
int *degrees;

float *w0;

const int sigmoid_table_size = 1024;
float *sigmoid_table;
const float SIGMOID_RESOLUTION = sigmoid_table_size / (SIGMOID_BOUND * 2.0f);

uint64_t rng_seed[2];

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

// http://xoroshiro.di.unimi.it/#shootout
uint64_t lrand() {
  const uint64_t s0 = rng_seed[0];
  uint64_t s1 = rng_seed[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  rng_seed[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
  rng_seed[1] = rotl(s1, 36);                   // c
  return result;
}

static inline double drand() {
  const union un {
    uint64_t i;
    double d;
  } a = {UINT64_C(0x3FF) << 52 | lrand() >> 12};
  return a.d - 1.0;
}

inline void *aligned_malloc(
  size_t size,
  size_t align) {
#ifndef _MSC_VER
void *result;
if (posix_memalign(&result, align, size)) result = 0;
#else
void *result = _aligned_malloc(size, align);
#endif
return result;
}

inline void aligned_free(void *ptr) {
#ifdef _MSC_VER
_aligned_free(ptr);
#else
free(ptr);
#endif
}

void init_sigmoid_table() {
  float x;
  sigmoid_table = static_cast<float *>(
    aligned_malloc((sigmoid_table_size + 1) * sizeof(float), DEFAULT_ALIGN));
  for (int k = 0; k != sigmoid_table_size; k++) {
    x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
    sigmoid_table[k] = 1 / (1 + exp(-x));
  }
}

float FastSigmoid(float x) {
  if (x > SIGMOID_BOUND)
    return 1;
  else if (x < -SIGMOID_BOUND)
    return 0;
  int k = (x + SIGMOID_BOUND) * SIGMOID_RESOLUTION;
  return sigmoid_table[k];
}

inline int irand(int min, int max) { return lrand() % (max - min) + min; }

inline int irand(int max) { return lrand() % max; }

inline int sample_neighbor(int node) {
  if (offsets[node] == offsets[node + 1])
    return -1;
  return edges[irand(offsets[node], offsets[node + 1])];
}

inline int sample_rw(int node) {
  int n2 = node;
  while (drand() < ppralpha) {
    int neighbor = sample_neighbor(n2);
    if (neighbor == -1)
      return n2;
    n2 = neighbor;
  }
  return n2;
}

int ArgPos(char *str, int argc, char **argv) {
  for (int a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        cout << "Argument missing for " << str << endl;
        exit(1);
      }
      return a;
    }
  return -1;
}

inline void update(float *w_s, float *w_t, int label, const float bias) {
  float score = -bias;
AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    score += w_s[c] * w_t[c];
  score = (label - FastSigmoid(score)) * global_lr;
AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_t[c] += score * w_s[c];
AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_s[c] += score * w_t[c];
}

void Train() {
#pragma omp parallel num_threads(n_threads)
  {
    const float nce_bias = log(nv);
    const float nce_bias_neg = log(nv / float(n_samples));
    int tid = omp_get_thread_num();
    ull last_ncount = 0;
    ull ncount = 0;
    float lr = global_lr;
#pragma omp barrier
    while (1) {
      if (ncount - last_ncount > 10000) {
        ull diff = ncount - last_ncount;
#pragma omp atomic
        step += diff;
        if (step > total_steps)
          break;
        if (tid == 0)
          if (!silent)
            cout << fixed << "\r Progress " << std::setprecision(2)
                 << step / (float)(total_steps + 1) * 100 << "%";
        last_ncount = ncount;
      }
      int n1 = irand(nv);
      int n2 = sample_rw(n1);
      update(&w0[n1 * n_hidden], &w0[n2 * n_hidden], 1, nce_bias);
      for (int i = 0; i < n_samples; i++) {
        int neg = irand(nv);
        update(&w0[n1 * n_hidden], &w0[neg * n_hidden], 0, nce_bias_neg);
      }
      ncount++;
    }
  }
}





// [[Rcpp::export]]
SEXP embeddings_verse(Rcpp::IntegerVector dgrmatrix_p, Rcpp::IntegerVector dgrmatrix_j,
                      int n_epochs = 100000) {
  step = 0;
  silent = false;
  n_threads = 1;
  global_lr = 0.0025f;
  //n_epochs = ;
  n_hidden = 128;
  n_samples = 3;
  ppralpha = 0.85f;
  
  // 
  // Set random nr
  //
  ull x = time(nullptr);
  for (int i = 0; i < 2; i++) {
    ull z = x += UINT64_C(0x9E3779B97F4A7C15);
    z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
    rng_seed[i] = z ^ z >> 31;
  }
  init_sigmoid_table();

  // 
  // Create edges and offsets
  //
  nv = (unsigned long long)(dgrmatrix_p.size()-1); // nodes
  ne = (unsigned long long)dgrmatrix_j.size();     // edges with links between nodes
  offsets = static_cast<int *>(aligned_malloc((nv + 1) * sizeof(int32_t), DEFAULT_ALIGN));
  edges   = static_cast<int *>(aligned_malloc(ne * sizeof(int32_t), DEFAULT_ALIGN));

  for (int i = 0; i < (int)(ne); i++)
    edges[i] = dgrmatrix_j[i];
  for (int i = 0; i < (int)(nv+1); i++)
    offsets[i] = dgrmatrix_p[i];

  //
  // Compute embeddings
  //
  w0 = static_cast<float *>(aligned_malloc(nv * n_hidden * sizeof(float), DEFAULT_ALIGN));
  for (int i = 0; i < (int)(nv * n_hidden); i++)
    w0[i] = drand() - 0.5;
  degrees = (int *)malloc(nv * sizeof(int));
  for (int i = 0; i < (int)nv; i++)
    degrees[i] = offsets[i + 1] - offsets[i];
  total_steps = n_epochs * (long long)nv;
  cout << "Total steps (mil): " << total_steps / 1000000. << endl;
  chrono::steady_clock::time_point begin = chrono::steady_clock::now();
  Train();
  chrono::steady_clock::time_point end = chrono::steady_clock::now();
  
  cout << endl
       << "Calculations took "
       << chrono::duration_cast<std::chrono::duration<float>>(end - begin)
  .count()
       << " s to run" << endl;

  //
  // Copy data and embeddings
  //
  Rcpp::IntegerVector _edges(ne);
  Rcpp::IntegerVector _offsets(nv+1);
  for (int i = 0; i < (int)ne; i++)
    _edges[i] = edges[i];
  for (int i = 0; i < (int)(nv+1); i++)
    _offsets[i] = offsets[i];
  
  /*
  Rcpp::NumericVector embeddings(nv * n_hidden);  
  for (int i = 0; i < (int)(nv * n_hidden); i++){
    embeddings[i] = w0[i];
  }
   */
  Rcpp::NumericMatrix embeddings(nv, n_hidden);
  for (int i = 0; i < nv; i++){
    for (int j = 0; j < n_hidden; j++){
      embeddings(i, j) = w0[i*nv + j];
    }
  }

  // Return
  Rcpp::List go = Rcpp::List::create(Rcpp::Named("vertices") = nv,
                                     Rcpp::Named("edges") = ne, 
                                     Rcpp::Named("dimension") = n_hidden, 
                                     Rcpp::Named("embedding") = embeddings,
                                     Rcpp::Named("edges_data") = _edges, 
                                     Rcpp::Named("offsets_data") = _offsets);
  return go;
}
