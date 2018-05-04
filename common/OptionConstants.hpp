#ifndef OPTION_CONSTANTS_HPP
#define OPTION_CONSTANTS_HPP

#include "CudaInterop.h"
#include "Options.hpp"
#include <algorithm>
#include <vector>
#include <cmath>

using namespace std;

namespace trinom
{

enum class SortType : char
{
    WIDTH = 'W',
    HEIGHT = 'H',
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
        auto T = options.Maturities.at(ind);
        auto termUnitsInYearCount = ceil((real)year / termUnit);
        auto termStepCount = options.TermStepCounts.at(ind);
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

    static void sortConstants(vector<OptionConstants> &v, const SortType sortType, const bool isTest = false)
    {
        switch (sortType)
        {
        case SortType::WIDTH:
            if (isTest)
                cout << "Sorting options by width first, height second" << endl;
            sort(v.begin(), v.end(), [](const OptionConstants &a, const OptionConstants &b) -> bool { return (a.width < b.width || (a.width == b.width && a.n < b.n)); });
            break;
        case SortType::HEIGHT:
            if (isTest)
                cout << "Sorting options by height first, width second" << endl;
            sort(v.begin(), v.end(), [](const OptionConstants &a, const OptionConstants &b) -> bool { return (a.n < b.n || (a.n == b.n && a.width < b.width)); });
            break;
        }
    }
};
}

#endif
