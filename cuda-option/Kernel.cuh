#ifndef CUDA_KERNEL_OPTION_CUH
#define CUDA_KERNEL_OPTION_CUH

#include "../cuda/CudaDomain.cuh"

using namespace trinom;

namespace cuda
{

/**
Base class for kernel arguments.
Important! Don't call defined pure virtual functions within your implementation.
**/
class KernelArgsBase
{
protected:
    real *res;
    real *QsAll;
    real *QsCopyAll;
    real *alphasAll;

public:

    KernelArgsBase(real *res, real *QsAll, real *QsCopyAll, real *alphasAll)
    {
        this->res = res;
        this->QsAll = QsAll;
        this->QsCopyAll = QsCopyAll;
        this->alphasAll = alphasAll;
    }
    
    __device__ inline int getIdx() const { return threadIdx.x + blockDim.x * blockIdx.x; }

    __device__ virtual void init(const CudaOptions &options) = 0;

    __device__ virtual void switchQs()
    {
        auto QsT = QsAll;
        QsAll = QsCopyAll;
        QsCopyAll = QsT;
    }

    __device__ virtual void fillQs(const int count, const int value) = 0;

    __device__ virtual void setQAt(const int index, const real value) = 0;

    __device__ virtual void setQCopyAt(const int index, const real value) = 0;

    __device__ virtual void setAlphaAt(const int index, const real value) = 0;

    __device__ virtual void setResult(const int jmax) = 0;

    __device__ virtual real getQAt(const int index) const = 0;

    __device__ virtual real getAlphaAt(const int index) const = 0;
};

template<class KernelArgsT>
__global__ void kernelOneOptionPerThread(const CudaOptions options, KernelArgsT args)
{
    auto idx = args.getIdx();

    // Out of options check
    if (idx >= options.N) return;

    OptionConstants c;
    computeConstants(c, options, idx);

    args.init(options);
    args.setQAt(c.jmax, one);
    args.setAlphaAt(0, getYieldAtYear(c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize));

    for (auto i = 1; i <= c.n; ++i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = args.getAlphaAt(i-1);
        real alpha_val = 0;

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j            
            
            auto expp1 = j == jhigh ? zero : args.getQAt(jind + 1) * exp(-(alpha + (j + 1) * c.dr) * c.dt);
            auto expm = args.getQAt(jind) * exp(-(alpha + j * c.dr) * c.dt);
            auto expm1 = j == -jhigh ? zero : args.getQAt(jind - 1) * exp(-(alpha + (j - 1) * c.dr) * c.dt);
            real Q;

            if (i == 1) {
                if (j == -jhigh) {
                    Q = computeJValue(j + 1, c.jmax, c.M, 3) * expp1;
                } else if (j == jhigh) {
                    Q = computeJValue(j - 1, c.jmax, c.M, 1) * expm1;
                } else {
                    Q = computeJValue(j, c.jmax, c.M, 2) * expm;
                }
            }
            else if (i <= c.jmax) {
                if (j == -jhigh) {
                    Q = computeJValue(j + 1, c.jmax, c.M, 3) * expp1;
                } else if (j == -jhigh + 1) {
                    Q = computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1;
                } else if (j == jhigh) {
                    Q = computeJValue(j - 1, c.jmax, c.M, 1) * expm1;
                } else if (j == jhigh - 1) {
                    Q = computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm;
                } else {
                    Q = computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1;
                }
            } else {
                if (j == -jhigh) {
                    Q = computeJValue(j, c.jmax, c.M, 3) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1;
                } else if (j == -jhigh + 1) {
                    Q = computeJValue(j - 1, c.jmax, c.M, 2) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1;
                            
                } else if (j == jhigh) {
                    Q = computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 1) * expm;
                } else if (j == jhigh - 1) {
                    Q = computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 2) * expp1;
                            
                } else {
                    Q = ((j == -jhigh + 2) ? computeJValue(j - 2, c.jmax, c.M, 1) * args.getQAt(jind - 2) * exp(-(alpha + (j - 2) * c.dr) * c.dt) : zero) +
                        computeJValue(j - 1, c.jmax, c.M, 1) * expm1 +
                        computeJValue(j, c.jmax, c.M, 2) * expm +
                        computeJValue(j + 1, c.jmax, c.M, 3) * expp1 +
                        ((j == jhigh - 2) ? computeJValue(j + 2, c.jmax, c.M, 3) * args.getQAt(jind + 2) * exp(-(alpha + (j + 2) * c.dr) * c.dt) : zero);
                }
            }
            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            args.setQCopyAt(jind, Q); 
            alpha_val += Q * exp(-j * c.dr * c.dt);
        }

        args.setAlphaAt(i, computeAlpha(alpha_val, i-1, c.dt, c.termUnit, options.YieldPrices, options.YieldTimeSteps, options.YieldSize));

        args.fillQs(c.width, 0);
        // Switch Qs
        args.switchQs();
    }
    
    // Backward propagation

    args.fillQs(c.width, 100); // initialize to 100$

    for (auto i = c.n - 1; i >= 0; --i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = args.getAlphaAt(i);
        auto isMaturity = i == ((int)(c.t / c.dt));

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j
            auto callExp = exp(-(alpha + j * c.dr) * c.dt);

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (computeJValue(j, c.jmax, c.M, 1) * args.getQAt(jind) +
                    computeJValue(j, c.jmax, c.M, 2) * args.getQAt(jind - 1) +
                    computeJValue(j, c.jmax, c.M, 3) * args.getQAt(jind - 2)) *
                        callExp;
            }
            else if (j == -c.jmax)
            {
                // Bottom edge branching
                res = (computeJValue(j, c.jmax, c.M, 1) * args.getQAt(jind + 2) +
                    computeJValue(j, c.jmax, c.M, 2) * args.getQAt(jind + 1) +
                    computeJValue(j, c.jmax, c.M, 3) * args.getQAt(jind)) *
                        callExp;
            }
            else
            {
                // Standard branching
                res = (computeJValue(j, c.jmax, c.M, 1) * args.getQAt(jind + 1) +
                    computeJValue(j, c.jmax, c.M, 2) * args.getQAt(jind) +
                    computeJValue(j, c.jmax, c.M, 3) * args.getQAt(jind - 1)) *
                        callExp;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            args.setQCopyAt(jind, computeCallValue(isMaturity, c, res));
        }

        args.fillQs(c.width, 0);
        // Switch Qs
        args.switchQs();
    }

    args.setResult(c.jmax);
}

}

#endif