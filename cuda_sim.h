// cuda_sim.h — mini-runtime CUDA sur CPU (aucun GPU ni nvcc requis).
//
// Objectif : compiler et EXÉCUTER, avec g++, les vrais kernels de Melanie_vNN.cpp (gpu_scan_kernel,
// gpu_selftest_kernel) ainsi que toute l'orchestration hôte (gpu_worker, gpu_neighbor_worker,
// auto-test GPU, double tampon...) sans modification : voir `-DMELANIE_GPU_SIM`.
//
// Modèle d'exécution (fidèle sur ce qui compte pour la mutualisation d'inversion par bloc) :
//   - un bloc = B "threads CUDA" qui s'exécutent comme des FIBRES (ucontext) coopératives, chacune avec
//     sa propre pile, ses propres variables locales, son propre threadIdx.x ;
//   - __shared__ = une zone commune à toutes les fibres du bloc (static thread_local : un bloc à la fois
//     par thread hôte) ;
//   - __syncthreads() = vraie barrière : une fibre qui l'atteint est suspendue tant que les B fibres du
//     bloc ne l'ont pas toutes atteinte. Si une fibre TERMINE alors que d'autres attendent à une
//     barrière (barrière divergente, comportement indéfini sur GPU), le simulateur s'arrête avec une
//     erreur explicite : c'est le bug typique d'un `return` anticipé avant un __syncthreads() ;
//   - les blocs sont répartis sur plusieurs threads hôtes (MELANIE_SIM_THREADS, défaut = nb de coeurs) ;
//   - les lancements sont synchrones (un ordonnancement légal pour des streams asynchrones) ;
//   - MELANIE_SIM_GPUS (défaut 1) = nombre de GPU simulés (pour tester le chemin multi-GPU).
// Non simulé : warps/lock-step (aucun intrinsic de warp n'est utilisé par le code), latences, occupation.
#pragma once
#include <ucontext.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

struct sim_dim3 { unsigned x = 0, y = 0, z = 0; };
inline thread_local sim_dim3 threadIdx, blockIdx, blockDim, gridDim;

namespace simcu {

enum : unsigned char { READY = 0, AT_BARRIER = 1, DONE = 2 };
static const size_t STACK_BYTES = 256 * 1024;

struct Block {
    ucontext_t sched;
    std::vector<ucontext_t> ctx;
    std::vector<std::unique_ptr<char[]>> stk;
    std::vector<unsigned char> st;
    const std::function<void()> *body = nullptr;
};
inline thread_local Block *cur_blk = nullptr;

[[noreturn]] inline void fatal(const char *msg, unsigned a, unsigned b) {
    fprintf(stderr, "\n[cuda_sim] ERREUR : %s (%u / %u) — bloc %u\n", msg, a, b, blockIdx.x);
    fflush(stderr);
    abort();
}

inline void entry() {
    Block *b = cur_blk;
    (*b->body)();
    b->st[threadIdx.x] = DONE; // au retour, uc_link reprend l'ordonnanceur
}

__attribute__((noinline)) inline void syncthreads() {
    Block *b = cur_blk;
    unsigned t = threadIdx.x;
    b->st[t] = AT_BARRIER;
    swapcontext(&b->ctx[t], &b->sched);
}

inline void run_block(unsigned bx, unsigned grid, unsigned B, const std::function<void()> &body) {
    static thread_local Block blk;
    if (blk.ctx.size() != B) {
        blk.ctx.assign(B, ucontext_t());
        blk.stk.clear();
        for (unsigned t = 0; t < B; t++) blk.stk.emplace_back(new char[STACK_BYTES]);
        blk.st.assign(B, READY);
    }
    blk.body = &body;
    cur_blk = &blk;
    blockIdx.x = bx; blockIdx.y = blockIdx.z = 0;
    gridDim.x = grid; gridDim.y = gridDim.z = 1;
    blockDim.x = B; blockDim.y = blockDim.z = 1;
    for (unsigned t = 0; t < B; t++) {
        getcontext(&blk.ctx[t]);
        blk.ctx[t].uc_stack.ss_sp = blk.stk[t].get();
        blk.ctx[t].uc_stack.ss_size = STACK_BYTES;
        blk.ctx[t].uc_link = &blk.sched;
        makecontext(&blk.ctx[t], entry, 0);
        blk.st[t] = READY;
    }
    unsigned ndone = 0;
    while (ndone < B) {
        unsigned nbar = 0;
        for (unsigned t = 0; t < B; t++) {
            if (blk.st[t] != READY) continue;
            threadIdx.x = t; threadIdx.y = threadIdx.z = 0;
            swapcontext(&blk.sched, &blk.ctx[t]);
            if (blk.st[t] == DONE) ndone++;
            else if (blk.st[t] == AT_BARRIER) nbar++;
        }
        if (nbar) {
            // Chaque fibre non terminée est maintenant soit DONE soit AT_BARRIER. Une barrière n'est
            // valide que si TOUTES les fibres du bloc y arrivent.
            if (ndone > 0) fatal("barriere divergente : des threads ont termine alors que d'autres attendent a __syncthreads()", ndone, nbar);
            for (unsigned t = 0; t < B; t++) if (blk.st[t] == AT_BARRIER) blk.st[t] = READY;
        }
    }
    cur_blk = nullptr;
}

template <class F>
inline void launch(unsigned grid, unsigned block, F &&kernel) {
    if (block == 0 || block > 1024) fatal("configuration de lancement invalide (threads par bloc)", block, 0);
    std::function<void()> body = [&]() { kernel(); };
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    if (const char *e = getenv("MELANIE_SIM_THREADS")) hw = std::max(1, atoi(e));
    unsigned nth = std::max(1u, std::min(grid, hw));
    std::atomic<unsigned> next{0};
    auto work = [&]() {
        for (;;) {
            unsigned b = next.fetch_add(1);
            if (b >= grid) break;
            run_block(b, grid, block, body);
        }
    };
    std::vector<std::thread> th;
    for (unsigned i = 1; i < nth; i++) th.emplace_back(work);
    work();
    for (auto &t : th) t.join();
}

} // namespace simcu

// ---------------------------------------------------------------------------------------------
// Mots-clés CUDA -> C++ ordinaire (définis APRÈS les #include standard du fichier principal)
// ---------------------------------------------------------------------------------------------
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static thread_local
#define __constant__ static const
#define __launch_bounds__(...)
inline void __syncthreads() { simcu::syncthreads(); }

inline int atomicAdd(int *p, int v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
inline unsigned long long atomicMax(unsigned long long *p, unsigned long long v) {
    unsigned long long o = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (o < v && !__atomic_compare_exchange_n(p, &o, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    return o;
}

// ---- runtime : mémoire, copies, streams (tout est synchrone) ----
enum cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorInvalidValue = 11 };
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
typedef struct simStream_st *cudaStream_t;

inline const char *cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "cuda_sim error"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int *n) {
    int v = 1;
    if (const char *e = getenv("MELANIE_SIM_GPUS")) v = atoi(e);
    *n = v;
    return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
template <class T> inline cudaError_t cudaMalloc(T **p, size_t n) {
    *p = (T *)malloc(n ? n : 1);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFree(void *p) { free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void *d, const void *s, size_t n, cudaMemcpyKind) { memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void *d, const void *s, size_t n, cudaMemcpyKind, cudaStream_t) { memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t *s) { *s = (cudaStream_t)new char; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t s) { delete (char *)s; return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }

// Lancement de kernel : nvcc -> k<<<g,b,0,s>>>(...) ; simulateur -> simcu::launch (voir le fichier principal).
#define GPU_LAUNCH(k, g, b, s, ...) simcu::launch((unsigned)(g), (unsigned)(b), [&]() { k(__VA_ARGS__); })
