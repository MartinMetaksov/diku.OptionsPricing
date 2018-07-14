#ifndef CUDA_DOMAIN_CUH
#define CUDA_DOMAIN_CUH

#include <chrono>

#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <limits>

#include "../common/Options.hpp"
#include "../common/OptionConstants.hpp"
#include "../common/Domain.hpp"
#include "../cuda/CudaErrors.cuh"

using namespace trinom;

namespace cuda
{

struct KernelOptions
{
    int N;
    int YieldSize;

    real *StrikePrices;
    real *Maturities;
    real *Lengths;
    uint16_t *TermUnits;
    uint16_t *TermStepCounts;
    real *ReversionRates;
    real *Volatilities;
    OptionType *Types;

    real *YieldPrices;
    int32_t *YieldTimeSteps;

    int32_t *Widths;
    int32_t *Heights;
};

struct compute_width_height
{
    template <typename Tuple>
    __host__ __device__ void operator()(Tuple t)
    {
        // Tuple(TermUnit, TermStepCount, Maturity, ReversionRate, Widths, Height)
        real termUnit = thrust::get<0>(t);
        real termStepCount = thrust::get<1>(t);
        real maturity = thrust::get<2>(t);
        real a = thrust::get<3>(t);
        int termUnitsInYearCount = ceil((real)year / termUnit);
        real dt = termUnitsInYearCount / termStepCount;               // [years]
        real M = exp(-a * dt) - one;
        int jmax = (int)(minus184 / M) + 1;

        thrust::get<4>(t) = 2 * jmax + 1;                                          // width
        thrust::get<5>(t) = termStepCount * termUnitsInYearCount * maturity + 1;   // height + 1
    }
};

struct sort_tuple_asc
{
    typedef thrust::tuple<int32_t, int32_t> Tuple;
    __host__ __device__ bool operator()(const Tuple& t1, const Tuple& t2)
    {
        return (t1.get<0>() < t2.get<0>() || (t1.get<0>() == t2.get<0>() && t1.get<1>() < t2.get<1>()));
    }
};

struct sort_tuple_desc
{
    typedef thrust::tuple<int32_t, int32_t> Tuple;
    __host__ __device__ bool operator()(const Tuple& t1, const Tuple& t2)
    {
        return (t1.get<0>() > t2.get<0>() || (t1.get<0>() == t2.get<0>() && t1.get<1>() > t2.get<1>()));
    }
};

template<typename T>
size_t vectorsizeof(const typename thrust::device_vector<T>& vec)
{
    return sizeof(T) * vec.size();
}

class CudaOptions
{
private:
    thrust::device_vector<real> StrikePrices;
    thrust::device_vector<real> Maturities;
    thrust::device_vector<real> Lengths;
    thrust::device_vector<uint16_t> TermUnits;
    thrust::device_vector<uint16_t> TermStepCounts;
    thrust::device_vector<real> ReversionRates;
    thrust::device_vector<real> Volatilities;
    thrust::device_vector<OptionType> Types;

    thrust::device_vector<real> YieldPrices;
    thrust::device_vector<int32_t> YieldTimeSteps;

public:
    const int N;
    const int YieldSize;
    KernelOptions KernelOptions;

    thrust::device_vector<int32_t> Widths;
    thrust::device_vector<int32_t> Heights;
    thrust::device_vector<int32_t> Indices;

    long DeviceMemory = 0;

    CudaOptions(const Options &options, const Yield &yield) : 
        
        StrikePrices(options.StrikePrices.begin(), options.StrikePrices.end()),
        Maturities(options.Maturities.begin(), options.Maturities.end()),
        Lengths(options.Lengths.begin(), options.Lengths.end()),
        TermUnits(options.TermUnits.begin(), options.TermUnits.end()),
        TermStepCounts(options.TermStepCounts.begin(), options.TermStepCounts.end()),
        ReversionRates(options.ReversionRates.begin(), options.ReversionRates.end()),
        Volatilities(options.Volatilities.begin(), options.Volatilities.end()),
        Types(options.Types.begin(), options.Types.end()),
        YieldPrices(yield.Prices.begin(), yield.Prices.end()),
        YieldTimeSteps(yield.TimeSteps.begin(), yield.TimeSteps.end()),
        N(options.N),
        YieldSize(yield.N)
    {

    }

    void initialize()
    {
        Widths.resize(N);
        Heights.resize(N);

        KernelOptions.N = N;
        KernelOptions.YieldSize = YieldSize, 
        KernelOptions.StrikePrices = thrust::raw_pointer_cast(StrikePrices.data());
        KernelOptions.Maturities = thrust::raw_pointer_cast(Maturities.data());
        KernelOptions.Lengths = thrust::raw_pointer_cast(Lengths.data());
        KernelOptions.TermUnits = thrust::raw_pointer_cast(TermUnits.data());
        KernelOptions.TermStepCounts = thrust::raw_pointer_cast(TermStepCounts.data());
        KernelOptions.ReversionRates = thrust::raw_pointer_cast(ReversionRates.data());
        KernelOptions.Volatilities = thrust::raw_pointer_cast(Volatilities.data());
        KernelOptions.Types = thrust::raw_pointer_cast(Types.data());
        KernelOptions.YieldPrices = thrust::raw_pointer_cast(YieldPrices.data());
        KernelOptions.YieldTimeSteps = thrust::raw_pointer_cast(YieldTimeSteps.data());
        KernelOptions.Widths = thrust::raw_pointer_cast(Widths.data());
        KernelOptions.Heights = thrust::raw_pointer_cast(Heights.data());

        DeviceMemory += vectorsizeof(StrikePrices);
        DeviceMemory += vectorsizeof(Maturities);
        DeviceMemory += vectorsizeof(Lengths);
        DeviceMemory += vectorsizeof(TermUnits);
        DeviceMemory += vectorsizeof(TermStepCounts);
        DeviceMemory += vectorsizeof(ReversionRates);
        DeviceMemory += vectorsizeof(Volatilities);
        DeviceMemory += vectorsizeof(Types);
        DeviceMemory += vectorsizeof(YieldPrices);
        DeviceMemory += vectorsizeof(YieldTimeSteps);
        DeviceMemory += vectorsizeof(Widths);
        DeviceMemory += vectorsizeof(Heights);

        // Fill in widths and heights for all options.
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(TermUnits.begin(), TermStepCounts.begin(), Maturities.begin(), ReversionRates.begin(), Widths.begin(), Heights.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(TermUnits.end(), TermStepCounts.end(), Maturities.end(), ReversionRates.end(), Widths.end(), Heights.end())),
                     compute_width_height());

        cudaDeviceSynchronize();
    }

    void sortOptions(const SortType sort, const bool isTest)
    {
        if (sort != SortType::NONE)
        {
            // Create indices
            Indices = thrust::device_vector<int32_t>(N);
            thrust::sequence(Indices.begin(), Indices.end());
            DeviceMemory += vectorsizeof(Indices);

            auto optionBegin = thrust::make_zip_iterator(thrust::make_tuple(StrikePrices.begin(), Maturities.begin(), Lengths.begin(), TermUnits.begin(), 
                TermStepCounts.begin(), ReversionRates.begin(), Volatilities.begin(), Types.begin(), Indices.begin()));
    
            auto keysBegin = (sort == SortType::WIDTH_ASC || sort == SortType::WIDTH_DESC) 
                ? thrust::make_zip_iterator(thrust::make_tuple(Widths.begin(), Heights.begin()))
                : thrust::make_zip_iterator(thrust::make_tuple(Heights.begin(), Widths.begin()));
            auto keysEnd = (sort == SortType::WIDTH_ASC || sort == SortType::WIDTH_DESC) 
                ? thrust::make_zip_iterator(thrust::make_tuple(Widths.end(), Heights.end()))
                : thrust::make_zip_iterator(thrust::make_tuple(Heights.end(), Widths.end()));

            // Sort options
            switch (sort)
            {
                case SortType::WIDTH_ASC:
                    if (isTest) std::cout << "Ascending sort, width first, height second" << std::endl;
                case SortType::HEIGHT_ASC:
                    if (isTest && sort == SortType::HEIGHT_ASC) std::cout << "Ascending sort, height first, width second" << std::endl;
                    thrust::sort_by_key(keysBegin, keysEnd, optionBegin, sort_tuple_asc());
                    break;
                case SortType::WIDTH_DESC:
                    if (isTest) std::cout << "Descending sort, width first, height second" << std::endl;
                case SortType::HEIGHT_DESC:
                    if (isTest && sort == SortType::HEIGHT_DESC) std::cout << "Descending sort, height first, width second" << std::endl;
                    thrust::sort_by_key(keysBegin, keysEnd, optionBegin, sort_tuple_desc());
                    break;
            }
            cudaDeviceSynchronize();
        }
    }

    void sortResult(thrust::device_vector<real> &deviceResults)
    {
        // Sort result
        if (!Indices.empty())
        {
            thrust::sort_by_key(Indices.begin(), Indices.end(), deviceResults.begin());
            cudaDeviceSynchronize();
        }
    }
};

struct CudaRuntime
{
    long KernelRuntime = std::numeric_limits<long>::max();
    long TotalRuntime = std::numeric_limits<long>::max();
    long DeviceMemory = 0;

};

bool operator <(const CudaRuntime& x, const CudaRuntime& y) {
    return std::tie(x.KernelRuntime, x.TotalRuntime) < std::tie(y.KernelRuntime, y.TotalRuntime);
}

__device__ void computeConstants(OptionConstants &c, const KernelOptions &options, const int idx)
{
    c.termUnit = options.TermUnits[idx];
    auto T = options.Maturities[idx];
    auto termUnitsInYearCount = ceil((real)year / c.termUnit);
    auto termStepCount = options.TermStepCounts[idx];
    c.t = options.Lengths[idx];
    c.n = termStepCount * termUnitsInYearCount * T;
    c.dt = termUnitsInYearCount / (real)termStepCount; // [years]
    c.type = options.Types[idx];

    auto a = options.ReversionRates[idx];
    c.X = options.StrikePrices[idx];
    auto sigma = options.Volatilities[idx];
    auto V = sigma * sigma * (one - exp(-two * a * c.dt)) / (two * a);
    c.dr = sqrt(three * V);
    c.M = exp(-a * c.dt) - one;

    // simplified computations
    // c.dr = sigma * sqrt(three * c.dt);
    // c.M = -a * c.dt;

    c.jmax = (int)(minus184 / c.M) + 1;
    c.width = 2 * c.jmax + 1;
}

}
#endif
