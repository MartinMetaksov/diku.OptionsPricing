#ifndef CUDA_OPTION_CUH
#define CUDA_OPTION_CUH

#define CUDA

#include "../common/Real.hpp"
#include "../common/OptionConstants.hpp"
#include "../common/Domain.hpp"
#include "../cuda/CudaErrors.cuh"
#include <cuda_runtime.h>
#include <chrono>

using namespace chrono;
using namespace trinom;

namespace cuda
{
    
__global__ void
computeSingleOptionKernel(real *res, OptionConstants *options, real *QsAll, real *QsCopyAll, real *alphasAll, int *QsInd, int *alphasInd, int totalCount)
{
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Out of options check
    if (idx >= totalCount) return;

    auto c = options[idx];
    auto Qs = QsAll + QsInd[idx];
    auto QsCopy = QsCopyAll + QsInd[idx];
    auto alphas = alphasAll + alphasInd[idx];

    // some test result
    res[idx] = c.n;
}

void computeOptions(OptionConstants *options, real *result, int count, bool isTest = false)
{
    // Compute indices
    int* QsInd = new int[count];
    int* alphasInd = new int[count];
    QsInd[0] = 0;
    alphasInd[0] = 0;
    int totalQsCount = 0;
    int totalAlphasCount = 0;
    for (auto i = 0; i < count - 1; ++i)
    {
        auto &option = options[i];
        totalQsCount += option.width;
        totalAlphasCount += option.n + 1;
        QsInd[i + 1] = totalQsCount;
        alphasInd[i + 1] = totalAlphasCount;
    }
    totalQsCount += options[count - 1].width;
    totalAlphasCount += options[count - 1].n + 1;
    
    auto blockSize = 32;
    auto blockCount = count / (blockSize + 1) + 1;

    if (isTest)
    {
        int memorySize = count * sizeof(real) + count * sizeof(OptionConstants) + 2 * count * sizeof(int)
                        + 2 * totalQsCount * sizeof(real) + totalAlphasCount * sizeof(real);
        cout << "Running trinomial option pricing for " << count << " options with block size " << blockSize << endl;
        cout << "Global memory size " << memorySize / (1024.0 * 1024.0) << " MB" << endl;
    }

    auto time_begin = steady_clock::now();

    real *d_result, *d_Qs, *d_QsCopy, *d_alphas;
    int *d_QsInd, *d_alphasInd;
    OptionConstants *d_options;
    CudaSafeCall(cudaMalloc((void **)&d_result, count * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_options, count * sizeof(OptionConstants)));
    CudaSafeCall(cudaMalloc((void **)&d_QsInd, count * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_alphasInd, count * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_Qs, totalQsCount * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_QsCopy, totalQsCount * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_alphas, totalAlphasCount * sizeof(real)));

    cudaMemcpy(d_options, options, count * sizeof(OptionConstants), cudaMemcpyHostToDevice);
    cudaMemcpy(d_QsInd, QsInd, count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alphasInd, alphasInd, count * sizeof(int), cudaMemcpyHostToDevice);

    auto time_begin_kernel = steady_clock::now();
    computeSingleOptionKernel<<<blockCount, blockSize>>>(d_result, d_options, d_Qs, d_QsCopy, d_alphas, d_QsInd, d_alphasInd, count);
    cudaThreadSynchronize();
    auto time_end_kernel = steady_clock::now();

    CudaCheckError();

    // Copy result
    cudaMemcpy(result, d_result, count * sizeof(real), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
    cudaFree(d_options);
    cudaFree(d_QsInd);
    cudaFree(d_alphasInd);
    cudaFree(d_Qs);
    cudaFree(d_QsCopy);
    cudaFree(d_alphas);

    auto time_end = steady_clock::now();
    if (isTest)
    {
        cout << "Kernel executed in " << duration_cast<milliseconds>(time_end_kernel - time_begin_kernel).count() << " ms" << endl;
        cout << "Total GPU time: " << duration_cast<milliseconds>(time_end - time_begin).count() << " ms" << endl
            << endl;
    }
}

}

#endif
