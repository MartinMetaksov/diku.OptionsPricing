#ifndef CUDA_VERSION_3_CUH
#define CUDA_VERSION_3_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"
#include <thrust/transform_scan.h>

namespace cuda
{

namespace option
{

struct KernelArgsValuesChunk
{
    real *res;
    real *QsAll;
    real *QsCopyAll;
    real *alphasAll;
    int32_t *QsInds;
    int32_t *alphasInds;
    int32_t chunkSize;
};

class KernelArgsCoalescedChunk : public KernelArgsBase<KernelArgsValuesChunk>
{
private:
    int widthStartIndex;
    int heightStartIndex;

    __device__ inline int getArrayIndex(int index) const { return values.chunkSize * index + threadIdx.x % values.chunkSize; }

public:

    KernelArgsCoalescedChunk(KernelArgsValuesChunk &v) : KernelArgsBase(v) { }

    __device__ inline void init(const CudaOptions &options) override
    {
        const int chunkIndex = getIdx() / values.chunkSize;
        widthStartIndex = chunkIndex == 0 ? 0 : values.QsInds[chunkIndex - 1];
        heightStartIndex = chunkIndex == 0 ? 0 : values.alphasInds[chunkIndex - 1];
    }

    __device__ void fillQs(const int count, const int value) override
    {
        auto ptr = values.QsAll + widthStartIndex + threadIdx.x % values.chunkSize;

        for (auto i = 0; i < count; ++i)
        {
            *ptr = value;
            ptr += values.chunkSize;
        }
    }

    __device__ inline void setQAt(const int index, const real value) override
    {
        values.QsAll[widthStartIndex + getArrayIndex(index)] = value;
    }

    __device__ inline void setQCopyAt(const int index, const real value) override
    {
        values.QsCopyAll[widthStartIndex + getArrayIndex(index)] = value;
    }

    __device__ inline void setAlphaAt(const int index, const real value) override
    {
        values.alphasAll[heightStartIndex + getArrayIndex(index)] = value;
    }

    __device__ inline void setResult(const int jmax) override
    {
        values.res[getIdx()] = values.QsAll[widthStartIndex + getArrayIndex(jmax)];
    }

    __device__ inline real getQAt(const int index) const override { return values.QsAll[widthStartIndex + getArrayIndex(index)]; }

    __device__ inline real getAlphaAt(const int index) const override { return values.alphasAll[heightStartIndex + getArrayIndex(index)]; }
};

struct same_chunk_indices
{
    const int32_t ChunkSize;

    same_chunk_indices(int32_t chunkSize) : ChunkSize(chunkSize) {}

    __host__ __device__ bool operator()(const int32_t &lhs, const int32_t &rhs) const {return lhs / ChunkSize == rhs / ChunkSize;}
};

struct times_chunk_size
{
    const int32_t ChunkSize;

    times_chunk_size(int32_t chunkSize) : ChunkSize(chunkSize) {}

    __host__ __device__ int32_t operator()(const int32_t &x) const {return x * ChunkSize;}
};

class KernelRunCoalescedChunk : public KernelRunBase
{
private:
    const int32_t ChunkSize;

public:
    KernelRunCoalescedChunk(int32_t chunkSize) : ChunkSize(chunkSize) { }

protected:
    void runPreprocessing(CudaOptions &cudaOptions, std::vector<real> &results,
        thrust::device_vector<int32_t> &widths, thrust::device_vector<int32_t> &heights) override
    {
        // Create block indices.
        const auto blockCount = ceil(cudaOptions.N / ((float)blockSize));
        thrust::device_vector<int32_t> keys(cudaOptions.N);
        thrust::sequence(keys.begin(), keys.end());

        const auto chunkCount = ceil(cudaOptions.N / ((float)ChunkSize));
        thrust::device_vector<int32_t> QsInds(chunkCount);
        thrust::device_vector<int32_t> alphasInds(chunkCount);
        thrust::device_vector<int32_t> keysOut(chunkCount);
        thrust::reduce_by_key(keys.begin(), keys.end(), widths.begin(), keysOut.begin(), QsInds.begin(), same_chunk_indices(ChunkSize), thrust::maximum<int32_t>());
        thrust::reduce_by_key(keys.begin(), keys.end(), heights.begin(), keysOut.begin(), alphasInds.begin(), same_chunk_indices(ChunkSize), thrust::maximum<int32_t>());
    
        thrust::transform_inclusive_scan(QsInds.begin(), QsInds.end(), QsInds.begin(), times_chunk_size(ChunkSize), thrust::plus<int32_t>());
        thrust::transform_inclusive_scan(alphasInds.begin(), alphasInds.end(), alphasInds.begin(), times_chunk_size(ChunkSize), thrust::plus<int32_t>());
    
        const int totalQsCount = QsInds[chunkCount - 1];
        const int totalAlphasCount = alphasInds[chunkCount - 1];

        KernelArgsValuesChunk values;
        values.chunkSize = ChunkSize;
        values.QsInds  = thrust::raw_pointer_cast(QsInds.data());
        values.alphasInds = thrust::raw_pointer_cast(alphasInds.data());

        runKernel<KernelArgsCoalescedChunk>(cudaOptions, results, totalQsCount, totalAlphasCount, values);
    }
};

}

}

#endif
