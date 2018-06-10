#ifndef OPTION_CONSTANTS_HPP
#define OPTION_CONSTANTS_HPP

#include <algorithm>
#include <vector>
#include <cmath>

#include "CudaInterop.h"
#include "Options.hpp"

namespace trinom
{

enum class SortType : char
{
    WIDTH_DESC = 'W',
    WIDTH_ASC = 'w',
    HEIGHT_DESC = 'H',
    HEIGHT_ASC = 'h',
    NONE = '-'
};

struct OptionConstants
{
    real t;
    real dt; // [years]
    real dr;
    real X;
    real M;
    int32_t jmax;
    int32_t n;
    int32_t width;
    uint16_t termUnit;
    OptionType type; // char

    DEVICE HOST OptionConstants() {}

    OptionConstants(const Options &options, const int ind)
    {
        termUnit = options.TermUnits.at(ind);
        const auto T = options.Maturities.at(ind);
        const int termUnitsInYearCount = ceil((real)year / termUnit);
        const auto termStepCount = options.TermStepCounts.at(ind);
        n = termStepCount * termUnitsInYearCount * T;
        t = options.Lengths.at(ind);
        dt = termUnitsInYearCount / (real)termStepCount; // [years]
        type = options.Types.at(ind);

        auto a = options.ReversionRates.at(ind);
        X = options.StrikePrices.at(ind);
        auto sigma = options.Volatilities.at(ind);
        auto V = sigma * sigma * (one - exp(-two * a * dt)) / (two * a);
        dr = sqrt(three * V);
        M = exp(-a * dt) - one;

        // simplified computations
        // dr = sigma * sqrt(three * dt);
        // M = -a * dt;

        jmax = (int)(minus184 / M) + 1;
        width = 2 * jmax + 1;
    }
};
} // namespace trinom

#endif
