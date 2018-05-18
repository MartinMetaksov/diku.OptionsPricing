#ifndef CUDA_VERSION_3_CUH
#define CUDA_VERSION_3_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"
#include <thrust/transform_scan.h>

using namespace chrono;
using namespace trinom;

namespace cuda
{

struct KernelArgsValuesBlock
{
    real *res;
    real *QsAll;
    real *QsCopyAll;
    real *alphasAll;
    int32_t *QsInds;
    int32_t *alphasInds;
};

class KernelArgsCoalescedBlock : public KernelArgsBase<KernelArgsValuesBlock>
{
private:
    int widthBlockStartIndex;
    int heightBlockStartIndex;

public:

    KernelArgsCoalescedBlock(KernelArgsValuesBlock &v) : KernelArgsBase(v) { }

    __device__ inline void init(const CudaOptions &options) override
    {
        widthBlockStartIndex = blockIdx.x == 0 ? 0 : values.QsInds[blockIdx.x - 1];
        heightBlockStartIndex = blockIdx.x == 0 ? 0 : values.alphasInds[blockIdx.x - 1];
    }

    __device__ void fillQs(const int count, const int value) override
    {
        auto ptr = values.QsAll + widthBlockStartIndex + threadIdx.x;

        for (auto i = 0; i < count; ++i)
        {
            *ptr = value;
            ptr += blockDim.x;
        }
    }

    __device__ inline void setQAt(const int index, const real value) override
    {
        values.QsAll[widthBlockStartIndex + blockDim.x * index + threadIdx.x] = value;
    }

    __device__ inline void setQCopyAt(const int index, const real value) override
    {
        values.QsCopyAll[widthBlockStartIndex + blockDim.x * index + threadIdx.x] = value;
    }

    __device__ inline void setAlphaAt(const int index, const real value) override
    {
        values.alphasAll[heightBlockStartIndex + blockDim.x * index + threadIdx.x] = value;
    }

    __device__ inline void setResult(const int jmax) override
    {
        values.res[getIdx()] = values.QsAll[widthBlockStartIndex + blockDim.x * jmax + threadIdx.x];
    }

    __device__ inline real getQAt(const int index) const override { return values.QsAll[widthBlockStartIndex + blockDim.x * index + threadIdx.x]; }

    __device__ inline real getAlphaAt(const int index) const override { return values.alphasAll[heightBlockStartIndex + blockDim.x * index + threadIdx.x]; }
};

struct same_block_indices
{
    const int32_t BlockSize;

    same_block_indices(int32_t blockSize) : BlockSize(blockSize) {}

    __host__ __device__ bool operator()(const int32_t &lhs, const int32_t &rhs) const {return lhs / BlockSize == rhs / BlockSize;}
};

struct times_block_size
{
    const int32_t BlockSize;

    times_block_size(int32_t blockSize) : BlockSize(blockSize) {}

    __host__ __device__ int32_t operator()(const int32_t &x) const {return x * BlockSize;}
};

class KernelRunCoalescedBlock : public KernelRunBase
{

protected:
    void runPreprocessing(CudaOptions &cudaOptions, vector<real> &results,
        thrust::device_vector<int32_t> &widths, thrust::device_vector<int32_t> &heights) override
    {
        // Create block indices.
        const auto blockCount = ceil(cudaOptions.N / ((float)blockSize));
        thrust::device_vector<int32_t> keys(cudaOptions.N);
        thrust::sequence(keys.begin(), keys.end());
    
        thrust::device_vector<int32_t> QsInds(blockCount);
        thrust::device_vector<int32_t> alphasInds(blockCount);
        thrust::device_vector<int32_t> keysOut(blockCount);
        thrust::reduce_by_key(keys.begin(), keys.end(), widths.begin(), keysOut.begin(), QsInds.begin(), same_block_indices(blockSize), thrust::maximum<int32_t>());
        thrust::reduce_by_key(keys.begin(), keys.end(), heights.begin(), keysOut.begin(), alphasInds.begin(), same_block_indices(blockSize), thrust::maximum<int32_t>());
    
        thrust::transform_inclusive_scan(QsInds.begin(), QsInds.end(), QsInds.begin(), times_block_size(blockSize), thrust::plus<int32_t>());
        thrust::transform_inclusive_scan(alphasInds.begin(), alphasInds.end(), alphasInds.begin(), times_block_size(blockSize), thrust::plus<int32_t>());
    
        const int totalQsCount = QsInds[blockCount - 1];
        const int totalAlphasCount = alphasInds[blockCount - 1];

        KernelArgsValuesBlock values;
        values.QsInds  = thrust::raw_pointer_cast(QsInds.data());
        values.alphasInds = thrust::raw_pointer_cast(alphasInds.data());

        runKernel<KernelArgsCoalescedBlock>(cudaOptions, results, totalQsCount, totalAlphasCount, values);
    }
};

}

#endif
