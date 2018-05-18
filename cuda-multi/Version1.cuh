#ifndef CUDA_MULTI_VERSION_1_CUH
#define CUDA_MULTI_VERSION_1_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

class KernelArgsNaive : public KernelArgsBase<KernelArgsValues>
{

public:

    KernelArgsNaive(KernelArgsValues &v) : KernelArgsBase(v) { }

};

struct same_chunk_indices
{
    const int32_t ChunkSize;

    same_chunk_indices(int32_t chunkSize) : ChunkSize(chunkSize) {}

    __host__ __device__ bool operator()(const int32_t &lhs, const int32_t &rhs) const {return lhs / ChunkSize == rhs / ChunkSize;}
};

class KernelRunNaive : public KernelRunBase
{

protected:
    void runPreprocessing(CudaOptions &cudaOptions, vector<real> &results,
        thrust::device_vector<int32_t> &widths, thrust::device_vector<int32_t> &heights) override
    {
        // Compute indices.
        thrust::host_vector<int32_t> hostWidths = widths;
        thrust::host_vector<int32_t> hInds;

        auto counter = 0;
        for (auto i = 0; i < cudaOptions.N; ++i)
        {
            auto w = hostWidths[i];
            counter += w;
            if (counter > blockSize)
            {
                hInds.push_back(i);
                counter = w;
            }
        }
        hInds.push_back(cudaOptions.N);

        thrust::device_vector<int32_t> dInds = hInds;

        KernelArgsValues values;

        runKernel<KernelArgsNaive>(cudaOptions, results, dInds, 0, values);
    }
};

}

#endif
