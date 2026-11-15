// Auto-test du simulateur lui-même : g++ -O2 -std=c++17 -pthread -o sim_selftest sim_selftest.cpp
//   ./sim_selftest          -> réduction par mémoire partagée + barrières, plusieurs blocs : doit afficher OK
//   ./sim_selftest diverge  -> barrière divergente volontaire : doit AVORTER avec un message explicite
#include <cstdio>
#include <cstring>
#include <vector>
#include "cuda_sim.h"

__global__ void reduce_kernel(const unsigned *in, unsigned *out, int bt) {
    __shared__ unsigned sh[256];
    unsigned t = threadIdx.x;
    sh[t] = in[blockIdx.x * blockDim.x + t];
    __syncthreads();
    for (unsigned s = bt / 2; s >= 1; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = sh[0];
}
__global__ void diverge_kernel(unsigned *out) {
    if (threadIdx.x == 3) return;      // sort AVANT la barrière : comportement indéfini sur GPU
    __syncthreads();
    out[threadIdx.x] = 1;
}

int main(int argc, char **argv) {
    const int BT = 128, GR = 9;
    if (argc > 1 && !strcmp(argv[1], "diverge")) {
        std::vector<unsigned> o(BT);
        GPU_LAUNCH(diverge_kernel, 2, BT, nullptr, o.data());
        printf("ERREUR : la barriere divergente n'a pas ete detectee\n");
        return 1;
    }
    std::vector<unsigned> in(BT * GR), out(GR, 0);
    for (size_t i = 0; i < in.size(); i++) in[i] = (unsigned)(i * 2654435761u) >> 20;
    GPU_LAUNCH(reduce_kernel, GR, BT, nullptr, in.data(), out.data(), BT);
    bool ok = true;
    for (int b = 0; b < GR; b++) {
        unsigned ref = 0;
        for (int t = 0; t < BT; t++) ref += in[b * BT + t];
        if (out[b] != ref) { ok = false; printf("bloc %d : %u != %u\n", b, out[b], ref); }
    }
    printf(ok ? "sim_selftest OK (reduction partagee, %d blocs x %d threads)\n" : "sim_selftest ECHEC\n", GR, BT);
    return ok ? 0 : 1;
}
