#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include "CudaInterop.h"
#include "Yield.hpp"
#include "OptionConstants.hpp"

using namespace std;

namespace trinom
{

DEVICE real getYieldAtYear(const real t, const int termUnit, const real *prices, const int32_t *timeSteps, const int size)
{
    const int tDays = (int)ROUND(t * termUnit);
    auto first = 0;
    auto second = 0;

    for (auto i = 0; i < size; ++i)
    {
        const auto t = timeSteps[i];
        if (t >= tDays)
        {
            second = i;
            break;
        }
        first = i;
    }

    // tDays <= timeSteps[0]
    if (first == second)
    {
        return prices[0];
    }

    // tDays > timeSteps[size-1]
    if (first == size - 1)
    {
        return prices[size - 1];
    }

    auto t1 = timeSteps[first];
    auto t2 = timeSteps[second];
    auto p1 = prices[first];
    auto p2 = prices[second];
    auto coefficient = (tDays - t1) / (real)(t2 - t1);
    return p1 + coefficient * (p2 - p1);
}

// Probability Equations

// Exhibit 1A (-jmax < j < jmax) (eq. 3A in Hull-White 1996)
DEVICE inline real PU_A(int j, real M)
{
    return one / six + (j * j * M * M + j * M) * half;
}

DEVICE inline real PM_A(int j, real M)
{
    return two / three - j * j * M * M;
}

DEVICE inline real PD_A(int j, real M)
{
    return one / six + (j * j * M * M - j * M) * half;
}

// Exhibit 1B (j == -jmax) (eq. 3C in Hull-White 1996)
DEVICE inline real PU_B(int j, real M)
{
    return one / six + (j * j * M * M - j * M) * half;
}

DEVICE inline real PM_B(int j, real M)
{
    return -one / three - j * j * M * M + two * j * M;
}

DEVICE inline real PD_B(int j, real M)
{
    return seven / six + (j * j * M * M - three * j * M) * half;
}

// Exhibit 1C (j == jmax) (eq. 3B in Hull-White 1996)
DEVICE inline real PU_C(int j, real M)
{
    return seven / six + (j * j * M * M + three * j * M) * half;
}

DEVICE inline real PM_C(int j, real M)
{
    return -one / three - j * j * M * M - two * j * M;
}

DEVICE inline real PD_C(int j, real M)
{
    return one / six + (j * j * M * M + j * M) * half;
}

DEVICE inline real computeAlpha(const real aggregatedQs, const int i, const real dt, const int termUnit, const real *prices, const int32_t *timeSteps, const int size)
{
    auto ti = (i + 2) * dt;
    auto R = getYieldAtYear(ti, termUnit, prices, timeSteps, size); // discount rate
    auto P = exp(-R * ti);                                          // discount bond price
    return log(aggregatedQs / P) / dt;                              // new alpha
}

DEVICE real computeJValue(const int j, const int jmax, const real M, const int expout)
{
    if (j == -jmax)
    {
        switch (expout)
        {
        case 1:
            return PU_B(j, M); // up
        case 2:
            return PM_B(j, M); // mid
        case 3:
            return PD_B(j, M); // down
        }
    }
    else if (j == jmax)
    {
        switch (expout)
        {
        case 1:
            return PU_C(j, M); // up
        case 2:
            return PM_C(j, M); // mid
        case 3:
            return PD_C(j, M); // down
        }
    }
    else
    {
        switch (expout)
        {
        case 1:
            return PU_A(j, M); // up
        case 2:
            return PM_A(j, M); // mid
        case 3:
            return PD_A(j, M); // down
        }
    }
    return 0;
}

DEVICE inline real computeCallValue(bool isMaturity, const OptionConstants &c, const real res)
{
    if (isMaturity)
    {
        switch (c.type)
        {
        case OptionType::PUT:
            return max(c.X - res, zero);
        case OptionType::CALL:
            return max(res - c.X, zero);
        }
    }
    return res;
}
} // namespace trinom

#endif
