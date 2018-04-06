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
    int n;
    real dt; // [years]
    real X;
    real dr;
    real M;
    int jmax;
    int width;
    OptionType type;

    static OptionConstants computeConstants(const Option &option)
    {
        OptionConstants c;
        auto T = option.Maturity;
        auto termUnitsInYearCount = ceil((real)year / option.TermUnit);
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
