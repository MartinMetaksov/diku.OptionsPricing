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

    static char* toStructureOfArrays(const vector<OptionConstants> &options)
    {
        const int count = options.size();
        char* array = new char[count * sizeof(OptionConstants)];

        real* ts = (real*)array;
        real* dts = ts + count;
        real* drs = dts + count;
        real* Xs = drs + count;
        real* Ms = Xs + count;
        int32_t* jmaxs = (int32_t*)(Ms + count);
        int32_t* ns = jmaxs + count;
        int32_t* widths = ns + count;
        uint16_t* termUnits = (uint16_t*)(widths + count);
        OptionType* types = (OptionType*)(termUnits + count);

        auto i = 0;
        for (auto &option : options)
        {
            ts[i] = option.t;
            dts[i] = option.dt;
            drs[i] = option.dr;
            Xs[i] = option.X;
            Ms[i] = option.M;
            jmaxs[i] = option.jmax;
            ns[i] = option.n;
            widths[i] = option.width;
            termUnits[i] = option.termUnit;
            types[i] = option.type;
            ++i;
        }
        return array;
    }
};
}

#endif
