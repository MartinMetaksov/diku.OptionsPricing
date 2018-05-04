#ifndef CUDA_DOMAIN_CUH
#define CUDA_DOMAIN_CUH

#include "../common/Options.hpp"
#include "../common/OptionConstants.hpp"
#include "../common/Domain.hpp"
#include "../cuda/CudaErrors.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <chrono>

using namespace trinom;

namespace cuda
{

struct compute_width_height
{
    template <typename Tuple>
    __host__ __device__ void operator()(Tuple t)
    {
        // Tuple(TermUnit, TermStepCount, Maturity, ReversionRate, Widths, Height)
        auto termUnit = t.get<0>();
        auto termStepCount = t.get<1>();
        auto maturity = t.get<2>();
        auto a = t.get<3>();
        auto termUnitsInYearCount = ceil((real)year / termUnit);
        auto dt = termUnitsInYearCount / (real)termStepCount;               // [years]
        auto M = exp(-a * dt) - one;
        auto jmax = (int)(minus184 / M) + 1;

        t.get<4>() = 2 * jmax + 1;                                          // width
        t.get<5>() = termStepCount * termUnitsInYearCount * maturity + 1;   // height + 1
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
    const uint16_t *StrikePrices;
    const uint16_t *Maturities;
    const uint16_t *Lengths;
    const uint16_t *TermUnits;
    const uint16_t *TermStepCounts;
    const real *ReversionRates;
    const real *Volatilities;
    const OptionType *Types;

    const real *YieldPrices;
    const int32_t *YieldTimeSteps;

    const int32_t *Widths;
    const int32_t *Heights;

    CudaOptions(
        const Options &options,
        const int yieldSize,
        const SortType sort,
        const bool isTest,
        thrust::device_vector<uint16_t> &strikePrices,
        thrust::device_vector<uint16_t> &maturities,
        thrust::device_vector<uint16_t> &lengths,
        thrust::device_vector<uint16_t> &termUnits,
        thrust::device_vector<uint16_t> &termStepCounts,
        thrust::device_vector<real> &reversionRates,
        thrust::device_vector<real> &volatilities,
        thrust::device_vector<OptionType> &types,
        thrust::device_vector<real> &yieldPrices,
        thrust::device_vector<int32_t> &yieldTimeSteps,
        thrust::device_vector<int32_t> &widths,
        thrust::device_vector<int32_t> &heights)
    {
        N = options.N;
        YieldSize = yieldSize;
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
                     thrust::make_zip_iterator(thrust::make_tuple(termUnits.end(), termStepCounts.end(), maturities.end(), reversionRates.end(), heights.end(), heights.end())),
                     compute_width_height());

        if (sort != SortType::NONE)
        {
            auto optionBegin = thrust::make_zip_iterator(thrust::make_tuple(strikePrices.begin(), maturities.begin(), lengths.begin(), termUnits.begin(), 
                termStepCounts.begin(), reversionRates.begin(), volatilities.begin(), types.begin(), yieldPrices.begin(), yieldTimeSteps.begin()));
    
            auto keysBegin = (sort == SortType::WIDTH_ASC || sort == SortType::WIDTH_DESC) 
                ? thrust::make_zip_iterator(thrust::make_tuple(widths.begin(), heights.begin()))
                : thrust::make_zip_iterator(thrust::make_tuple(heights.begin(), widths.begin()));
            auto keysEnd = (sort == SortType::WIDTH_ASC || sort == SortType::WIDTH_DESC) 
                ? thrust::make_zip_iterator(thrust::make_tuple(widths.end(), heights.end()))
                : thrust::make_zip_iterator(thrust::make_tuple(heights.end(), widths.end()));

            switch (sort)
            {
                case SortType::WIDTH_ASC:
                    if (isTest) cout << "Ascending sort, width first, height second" << endl;
                case SortType::HEIGHT_ASC:
                    if (isTest && sort == SortType::HEIGHT_ASC) cout << "Ascending sort, height first, width second" << endl;
                    thrust::sort_by_key(keysBegin, keysEnd, optionBegin, sort_tuple_asc());
                    break;
                case SortType::WIDTH_DESC:
                    if (isTest) cout << "Descending sort, width first, height second" << endl;
                case SortType::HEIGHT_DESC:
                    if (isTest && sort == SortType::HEIGHT_DESC) cout << "Descending sort, height first, width second" << endl;
                    thrust::sort_by_key(keysBegin, keysEnd, optionBegin, sort_tuple_desc());
                    break;
            }
        }
    }
};

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
