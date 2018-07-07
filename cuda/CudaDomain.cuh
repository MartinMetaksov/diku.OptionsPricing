#ifndef CUDA_DOMAIN_CUH
#define CUDA_DOMAIN_CUH

#include <chrono>

#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>

#include "../common/Options.hpp"
#include "../common/OptionConstants.hpp"
#include "../common/Domain.hpp"
#include "../cuda/CudaErrors.cuh"

using namespace trinom;

namespace cuda
{

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

struct CudaOptions
{
    int N;
    int YieldSize;
    SortType Sort;

    const real *StrikePrices;
    const real *Maturities;
    const real *Lengths;
    const uint16_t *TermUnits;
    const uint16_t *TermStepCounts;
    const real *ReversionRates;
    const real *Volatilities;
    const OptionType *Types;

    const real *YieldPrices;
    const int32_t *YieldTimeSteps;

    const int32_t *Widths;
    const int32_t *Heights;
    thrust::device_vector<int32_t>::iterator IndicesBegin;
    thrust::device_vector<int32_t>::iterator IndicesEnd;

    CudaOptions(
        const Options &options,
        const int yieldSize,
        const SortType sort,
        const bool isTest,
        thrust::device_vector<real> &strikePrices,
        thrust::device_vector<real> &maturities,
        thrust::device_vector<real> &lengths,
        thrust::device_vector<uint16_t> &termUnits,
        thrust::device_vector<uint16_t> &termStepCounts,
        thrust::device_vector<real> &reversionRates,
        thrust::device_vector<real> &volatilities,
        thrust::device_vector<OptionType> &types,
        thrust::device_vector<real> &yieldPrices,
        thrust::device_vector<int32_t> &yieldTimeSteps,
        thrust::device_vector<int32_t> &widths,
        thrust::device_vector<int32_t> &heights,
        thrust::device_vector<int32_t> &indices)
    {
        N = options.N;
        YieldSize = yieldSize;
        Sort = sort;
        StrikePrices = thrust::raw_pointer_cast(strikePrices.data());
        Maturities = thrust::raw_pointer_cast(maturities.data());
        Lengths = thrust::raw_pointer_cast(lengths.data());
        TermUnits = thrust::raw_pointer_cast(termUnits.data());
        TermStepCounts = thrust::raw_pointer_cast(termStepCounts.data());
        ReversionRates = thrust::raw_pointer_cast(reversionRates.data());
        Volatilities = thrust::raw_pointer_cast(volatilities.data());
        Types = thrust::raw_pointer_cast(types.data());
        YieldPrices = thrust::raw_pointer_cast(yieldPrices.data());
        YieldTimeSteps = thrust::raw_pointer_cast(yieldTimeSteps.data());
        Widths = thrust::raw_pointer_cast(widths.data());
        Heights = thrust::raw_pointer_cast(heights.data());

        // Fill in widths and heights for all options.
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(termUnits.begin(), termStepCounts.begin(), maturities.begin(), reversionRates.begin(), widths.begin(), heights.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(termUnits.end(), termStepCounts.end(), maturities.end(), reversionRates.end(), widths.end(), heights.end())),
                     compute_width_height());

        if (sort != SortType::NONE)
        {
            // Create indices
            IndicesBegin = indices.begin();
            IndicesEnd = indices.end();
            thrust::sequence(IndicesBegin, IndicesEnd);

            auto optionBegin = thrust::make_zip_iterator(thrust::make_tuple(strikePrices.begin(), maturities.begin(), lengths.begin(), termUnits.begin(), 
                termStepCounts.begin(), reversionRates.begin(), volatilities.begin(), types.begin(), indices.begin()));
    
            auto keysBegin = (sort == SortType::WIDTH_ASC || sort == SortType::WIDTH_DESC) 
                ? thrust::make_zip_iterator(thrust::make_tuple(widths.begin(), heights.begin()))
                : thrust::make_zip_iterator(thrust::make_tuple(heights.begin(), widths.begin()));
            auto keysEnd = (sort == SortType::WIDTH_ASC || sort == SortType::WIDTH_DESC) 
                ? thrust::make_zip_iterator(thrust::make_tuple(widths.end(), heights.end()))
                : thrust::make_zip_iterator(thrust::make_tuple(heights.end(), widths.end()));

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
        }
        cudaDeviceSynchronize();
    }

    void sortResult(thrust::device_vector<real> &deviceResults)
    {
        // Sort result
        if (Sort != SortType::NONE)
        {
            thrust::sort_by_key(IndicesBegin, IndicesEnd, deviceResults.begin());
            cudaDeviceSynchronize();
        }
    }
};

template<typename T>
size_t vectorsizeof(const typename thrust::device_vector<T>& vec)
{
    return sizeof(T) * vec.size();
}

struct CudaRuntime
{
    long KernelRuntime;
    long TotalRuntime;
};

bool operator <(const CudaRuntime& x, const CudaRuntime& y) {
    return std::tie(x.KernelRuntime, x.TotalRuntime) < std::tie(y.KernelRuntime, y.TotalRuntime);
}

__device__ void computeConstants(OptionConstants &c, const CudaOptions &options, const int idx)
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
