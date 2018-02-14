#include "Real.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

// Follows code independent of the instantiation of real
const real zero = 0;
const real one = 1;
const real two = 2;
const real half = one / two;
const real three = 3;
const real six = 6;
const real seven = 7;
const real year = 365;
const real minus184 = -0.184;

struct Yield
{
    real p;
    real t;
};

#ifdef CUDA
__constant__
#endif
    Yield h_YieldCurve[] =
        {{.p = 0.0501772, .t = 3.0},    //
         {.p = 0.0509389, .t = 367.0},  //
         {.p = 0.0579733, .t = 731.0},  //
         {.p = 0.0630595, .t = 1096.0}, //
         {.p = 0.0673464, .t = 1461.0}, //
         {.p = 0.0694816, .t = 1826.0}, //
         {.p = 0.0708807, .t = 2194.0}, //
         {.p = 0.0727527, .t = 2558.0}, //
         {.p = 0.0730852, .t = 2922.0}, //
         {.p = 0.0739790, .t = 3287.0}, //
         {.p = 0.0749015, .t = 3653.0}};

// Probability Equations

// Exhibit 1A (-jmax < j < jmax)
#ifdef CUDA
__device__
#endif
    inline real
    PU_A(int j, real M)
{
    return one / six + (((real)(j * j)) * M * M + ((real)j) * M) * half;
}

#ifdef CUDA
__device__
#endif
    inline real
    PM_A(int j, real M)
{
    return two / three - (((real)(j * j)) * M * M);
}

#ifdef CUDA
__device__
#endif
    inline real
    PD_A(int j, real M)
{
    return one / six + (((real)(j * j)) * M * M - ((real)j) * M) * half;
}

// Exhibit 1B (j == -jmax)
#ifdef CUDA
__device__
#endif
    inline real
    PU_B(int j, real M)
{
    return one / six + (((real)(j * j)) * M * M - ((real)j) * M) * half;
}

#ifdef CUDA
__device__
#endif
    inline real
    PM_B(int j, real M)
{
    return -one / three - (((real)(j * j)) * M * M) + two * ((real)j) * M;
}

#ifdef CUDA
__device__
#endif
    inline real
    PD_B(int j, real M)
{
    return seven / six + (((real)(j * j)) * M * M - three * ((real)j) * M) * half;
}

// Exhibit 1C (j == jmax)
#ifdef CUDA
__device__
#endif
    inline real
    PU_C(int j, real M)
{
    return seven / six + (((real)(j * j)) * M * M + three * ((real)j) * M) * half;
}

#ifdef CUDA
__device__
#endif
    inline real
    PM_C(int j, real M)
{
    return -one / three - (((real)(j * j)) * M * M) - two * ((real)j) * M;
}

#ifdef CUDA
__device__
#endif
    inline real
    PD_C(int j, real M)
{
    return one / six + (((real)(j * j)) * M * M + ((real)j) * M) * half;
}

// forward propagation helper
#ifdef CUDA
__device__
#endif
    inline real
    fwdHelper(int idx, real M, real dr, real dt, real alphai, volatile real *QCopy, int beg_ind, int m, int i, int imax, int jmax, int j)
{
    auto eRdt_u1 = exp(-(((real)(j + 1)) * dr + alphai) * dt);
    auto eRdt = exp(-(((real)j) * dr + alphai) * dt);
    auto eRdt_d1 = exp(-(((real)(j - 1)) * dr + alphai) * dt);
    if (i < jmax)
    {
        auto pu = PU_A(j - 1, M);
        auto pm = PM_A(j, M);
        auto pd = PD_A(j + 1, M);
        if (i == 0 && j == 0)
            return pm * QCopy[beg_ind + j + m] * eRdt;
        else if (j == -imax + 1)
            return pd * QCopy[beg_ind + j + m + 1] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt;
        else if (j == imax - 1)
            return pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j + m - 1] * eRdt_d1;
        else if (j == 0 - imax)
            return pd * QCopy[beg_ind + j + m + 1] * eRdt_u1;
        else if (j == imax)
            return pu * QCopy[beg_ind + j + m - 1] * eRdt_d1;
        else
            return pd * QCopy[beg_ind + j + m + 1] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j + m - 1] * eRdt_d1;
    }
    else if (j == jmax)
    {
        auto pm = PU_C(j, M);
        auto pu = PU_A(j - 1, M);
        return pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j - 1 + m] * eRdt_d1;
    }
    else if (j == jmax - 1)
    {
        auto pd = PM_C(j + 1, M);
        auto pm = PM_A(j, M);
        auto pu = PU_A(j - 1, M);
        return pd * QCopy[beg_ind + j + 1 + m] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j - 1 + m] * eRdt_d1;
    }
    else if (j == jmax - 2)
    {
        auto eRdt_u2 = exp(-(((real)(j + 2)) * dr + alphai) * dt);
        auto pd_c = PD_C(j + 2, M);
        auto pd = PD_A(j + 1, M);
        auto pm = PM_A(j, M);
        auto pu = PU_A(j - 1, M);
        return pd_c * QCopy[beg_ind + j + 2 + m] * eRdt_u2 + pd * QCopy[beg_ind + j + 1 + m] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j - 1 + m] * eRdt_d1;
    }
    else if (j == -jmax + 2)
    {
        auto eRdt_d2 = exp(-(((real)(j - 2)) * dr + alphai) * dt);
        auto pd = PD_A(j + 1, M);
        auto pm = PM_A(j, M);
        auto pu = PU_A(j - 1, M);
        auto pu_b = PU_B(j - 2, M);
        return pd * QCopy[beg_ind + j + 1 + m] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j - 1 + m] * eRdt_d1 + pu_b * QCopy[beg_ind + j - 2 + m] * eRdt_d2;
    }
    else if (j == -jmax + 1)
    {
        auto pd = PD_A(j + 1, M);
        auto pm = PM_A(j, M);
        auto pu = PM_B(j - 1, M);
        return pd * QCopy[beg_ind + j + 1 + m] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j - 1 + m] * eRdt_d1;
    }
    else if (j == -jmax)
    {
        auto pd = PD_A(j + 1, M);
        auto pm = PD_B(j, M);
        return pd * QCopy[beg_ind + j + 1 + m] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt;
    }
    else
    {
        auto pd = PD_A(j + 1, M);
        auto pm = PM_A(j, M);
        auto pu = PU_A(j - 1, M);
        return pd * QCopy[beg_ind + j + 1 + m] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j - 1 + m] * eRdt_d1;
    }
}

// backward propagation helper
#ifdef CUDA
__device__
#endif
    inline real
    bkwdHelper(real X, real M, real dr, real dt, real alphai, volatile real *CallCopy, int beg_ind, int m, int i, int jmax, int j)
{
    auto eRdt = exp(-(((real)j) * dr + alphai) * dt);
    real res;
    if (i < jmax)
    {
        // -- central node
        auto pu = PU_A(j, M);
        auto pm = PM_A(j, M);
        auto pd = PD_A(j, M);
        res = (pu * CallCopy[beg_ind + j + m + 1] + pm * CallCopy[beg_ind + j + m] + pd * CallCopy[beg_ind + j + m - 1]) * eRdt;
    }
    else if (j == jmax)
    {
        // top node
        auto pu = PU_C(j, M);
        auto pm = PM_C(j, M);
        auto pd = PD_C(j, M);
        res = (pu * CallCopy[beg_ind + j + m] + pm * CallCopy[beg_ind + j + m - 1] + pd * CallCopy[beg_ind + j + m - 2]) * eRdt;
    }
    else if (j == -jmax)
    {
        // bottom node
        auto pu = PU_B(j, M);
        auto pm = PM_B(j, M);
        auto pd = PD_B(j, M);
        res = (pu * CallCopy[beg_ind + j + m + 2] + pm * CallCopy[beg_ind + j + m + 1] + pd * CallCopy[beg_ind + j + m]) * eRdt;
    }
    else
    {
        // central node
        auto pu = PU_A(j, M);
        auto pm = PM_A(j, M);
        auto pd = PD_A(j, M);
        res = (pu * CallCopy[beg_ind + j + m + 1] + pm * CallCopy[beg_ind + j + m] + pd * CallCopy[beg_ind + j + m - 1]) * eRdt;
    }

    // TODO (WMP) This should be parametrized; length of contract, here 3 years
    if (i == ((int)(three / dt)))
        return max(X - res, zero);
    else
        return res;
}
