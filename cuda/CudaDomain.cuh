#ifndef CUDA_DOMAIN_CUH
#define CUDA_DOMAIN_CUH

#include "../common/Options.hpp"
#include "../common/OptionConstants.hpp"
#include "../common/Domain.hpp"
#include "../cuda/CudaErrors.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
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
        auto termUnit = thrust::get<0>(t);
        auto termStepCount = thrust::get<1>(t);
        auto maturity = thrust::get<2>(t);
        auto a = thrust::get<3>(t);
        auto termUnitsInYearCount = ceil((real)year / termUnit);
        auto dt = termUnitsInYearCount / (real)termStepCount;                       // [years]
        auto M = exp(-a * dt) - one;
        auto jmax = (int)(minus184 / M) + 1;

        thrust::get<4>(t) = 2 * jmax + 1;                                           // width
        thrust::get<5>(t) = termStepCount * termUnitsInYearCount * maturity + 1;    // height + 1
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
        const thrust::device_vector<uint16_t> &strikePrices,
        const thrust::device_vector<uint16_t> &maturities,
        const thrust::device_vector<uint16_t> &lengths,
        const thrust::device_vector<uint16_t> &termUnits,
        const thrust::device_vector<uint16_t> &termStepCounts,
        const thrust::device_vector<real> &reversionRates,
        const thrust::device_vector<real> &volatilities,
        const thrust::device_vector<OptionType> &types,
        const thrust::device_vector<real> &yieldPrices,
        const thrust::device_vector<int32_t> &yieldTimeSteps,
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
