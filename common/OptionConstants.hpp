#ifndef OPTION_CONSTANTS_HPP
#define OPTION_CONSTANTS_HPP

#include "Option.hpp"
#include <cmath>

using namespace std;

namespace trinom
{

struct OptionConstants
{
  public:
    real t;
    real dt; // [years]
    real dr;
    real X;
    real M;
    int32_t jmax;
    int32_t n;
    int32_t width;
    uint16_t termUnit;
    OptionType type;    // char
    uint8_t padding;

    static OptionConstants computeConstants(const Option &option)
    {
        OptionConstants c;
        auto T = option.Maturity;
        auto termUnitsInYearCount = ceil((real)year / option.TermUnit);
        c.termUnit = option.TermUnit;
        c.t = option.Length;
        c.n = option.TermStepCount * termUnitsInYearCount * T;
        c.dt = termUnitsInYearCount / (real)option.TermStepCount; // [years]
        c.type = option.Type;

        auto a = option.ReversionRate;
        c.X = option.StrikePrice;
        auto sigma = option.Volatility;
        auto V = sigma * sigma * (one - exp(-two * a * c.dt)) / (two * a);
        c.dr = sqrt(three * V);
        c.M = exp(-a * c.dt) - one;

        // simplified computations
        // c.dr = sigma * sqrt(three * c.dt);
        // c.M = -a * c.dt;

        c.jmax = (int)(minus184 / c.M) + 1;
        c.width = 2 * c.jmax + 1;
        return c;
    }
};
}

#endif
