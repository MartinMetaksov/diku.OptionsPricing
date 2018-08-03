#ifndef CUDA_VERSION_3_CUH
#define CUDA_VERSION_3_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

namespace option
{

struct KernelArgsValuesGranular
{
    real *res;
    real *QsAll;
    real *QsCopyAll;
    real *alphasAll;
    int32_t *QsInds;
    int32_t *alphasInds;
    int32_t granularity;
};

class KernelArgsCoalescedGranular : public KernelArgsBase<KernelArgsValuesGranular>
{
private:
    int widthStartIndex;
    int heightStartIndex;

    __device__ inline int getArrayIndex(int index) const { return values.granularity * index + threadIdx.x % values.granularity; }

public:

    KernelArgsCoalescedGranular(KernelArgsValuesGranular &v) : KernelArgsBase(v) { }

    __device__ inline void init(const KernelOptions &options) override
    {
        const int index = getIdx() / values.granularity;
        widthStartIndex = index == 0 ? 0 : values.QsInds[index - 1];
        heightStartIndex = index == 0 ? 0 : values.alphasInds[index - 1];
    }

    __device__ void fillQs(const int count, const int value) override
    {
        auto ptr = values.QsAll + widthStartIndex + threadIdx.x % values.granularity;

        for (auto i = 0; i < count; ++i)
        {
            *ptr = value;
            ptr += values.granularity;
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

struct same_granularity_indices
{
    const int32_t Granularity;

    same_granularity_indices(int32_t granularity) : Granularity(granularity) {}

    __host__ __device__ bool operator()(const int32_t &lhs, const int32_t &rhs) const {return lhs / Granularity == rhs / Granularity;}
};

struct times_granularity
{
    const int32_t Granularity;

    times_granularity(int32_t granularity) : Granularity(granularity) {}

    __host__ __device__ int32_t operator()(const int32_t &x) const {return x * Granularity;}
};

class KernelRunCoalescedGranular : public KernelRunBase
{
private:
    int32_t Granularity;

public:
    KernelRunCoalescedGranular(int32_t granularity) : Granularity(granularity) { }

protected:
    void runPreprocessing(CudaOptions &options, std::vector<real> &results) override
    {
        // Create block indices.
        thrust::device_vector<int32_t> keys(options.N);
        thrust::sequence(keys.begin(), keys.end());

        if (Granularity == -1) Granularity = BlockSize;
        const auto count = ceil(options.N / ((float)Granularity));
        thrust::device_vector<int32_t> QsInds(count);
        thrust::device_vector<int32_t> alphasInds(count);
        thrust::device_vector<int32_t> keysOut(count);
        thrust::reduce_by_key(keys.begin(), keys.end(), options.Widths.begin(), keysOut.begin(), QsInds.begin(), same_granularity_indices(Granularity), thrust::maximum<int32_t>());
        thrust::reduce_by_key(keys.begin(), keys.end(), options.Heights.begin(), keysOut.begin(), alphasInds.begin(), same_granularity_indices(Granularity), thrust::maximum<int32_t>());
    
        thrust::transform_inclusive_scan(QsInds.begin(), QsInds.end(), QsInds.begin(), times_granularity(Granularity), thrust::plus<int32_t>());
        thrust::transform_inclusive_scan(alphasInds.begin(), alphasInds.end(), alphasInds.begin(), times_granularity(Granularity), thrust::plus<int32_t>());
    
        const int totalQsCount = QsInds[count - 1];
        const int totalAlphasCount = alphasInds[count - 1];

        KernelArgsValuesGranular values;
        values.granularity = Granularity;
        values.QsInds  = thrust::raw_pointer_cast(QsInds.data());
        values.alphasInds = thrust::raw_pointer_cast(alphasInds.data());

        options.DeviceMemory += vectorsizeof(keys);
        options.DeviceMemory += vectorsizeof(QsInds);
        options.DeviceMemory += vectorsizeof(alphasInds);
        options.DeviceMemory += vectorsizeof(keysOut);

        runKernel<KernelArgsCoalescedGranular>(options, results, totalQsCount, totalAlphasCount, values);
    }
};

}

}

#endif
