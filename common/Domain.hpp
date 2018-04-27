#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include "Yield.hpp"
#include "OptionConstants.hpp"

using namespace std;

namespace trinom
{

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#else
#define CONSTANT const
#endif

#ifdef __CUDA_ARCH__
#define DEVICE __device__
#else
#define DEVICE
#endif

DEVICE real getYieldAtYear(const real t, const int termUnit, const Yield *curve, const int size)
{
    const int tDays = (int)ROUND(t * termUnit);
    auto first = curve[0];
    auto second = curve[0];

    for (auto i = 0; i < size; ++i)
    {
        const auto yield = curve[i];
        if (yield.t >= tDays)
        {
            second = yield;
            break;
        }
        first = yield;
    }

    // Prevent division by zero
    if (first.t == second.t)
    {
        return first.p;
    }

    auto coefficient = (tDays - first.t) / (real)(second.t - first.t);
    return first.p + coefficient * (second.p - first.p);
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

DEVICE inline real computeAlpha(const real aggregatedQs, const int i, const real dt, const int termUnit, const Yield *curve, const int size)
{
    auto ti = (i + 2) * dt;
    auto R = getYieldAtYear(ti, termUnit, curve, size); // discount rate
    auto P = exp(-R * ti);                              // discount bond price
    return log(aggregatedQs / P) / dt;                  // new alpha
}

DEVICE real computeJValue(const int i, const real dr, const real M, const int width, const int jmax, const int expout)
{
    // this i is only local and has nothing to do with the height of the tree.
    if (i == 0)
    {
        switch (expout)
        {
        case 1:
            return PU_B((-jmax), M); // up
        case 2:
            return PM_B((-jmax), M); // mid
        case 3:
            return PD_B((-jmax), M); // down
        case 0:
        default:
            return (-jmax) * dr; // rate
        }
    }
    else if (i == width - 1)
    {
        switch (expout)
        {
        case 1:
            return PU_C(jmax, M); // up
        case 2:
            return PM_C(jmax, M); // mid
        case 3:
            return PD_C(jmax, M); // down
        case 0:
        default:
            return jmax * dr; // rate
        }
    }
    else
    {
        auto j = i + (-jmax);
        switch (expout)
        {
        case 1:
            return PU_A(j, M); // up
        case 2:
            return PM_A(j, M); // mid
        case 3:
            return PD_A(j, M); // down
        case 0:
        default:
            return j * dr; // rate
        }
    }
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
    return isMaturity ? max(c.X - res, zero) : res;
}
}

#endif
