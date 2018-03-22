#define CUDA
#include "../common/Real.hpp"
#include "../common/OptionConstants.hpp"
#include "../common/FutharkArrays.hpp"
#include "../common/Domain.hpp"
#include "../cuda/CudaErrors.cuh"
#include "../cuda/ScanKernels.cuh"

#include <chrono>

using namespace std;
using namespace chrono;

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

struct jvalue
{
    real pu;
    real pm;
    real pd;
};

__global__ void
computeSingleOptionKernel(real *res, OptionConstants *options, int n_max)
{
    extern __shared__ char sh_mem[];

    volatile jvalue *jvalues = (jvalue *)&sh_mem;
    volatile real *Qs = (real *)&jvalues[blockDim.x];
    volatile real *QsCopy = &Qs[blockDim.x];
    volatile real *alphas = &QsCopy[blockDim.x];

    auto c = options[blockIdx.x];

    // Compute jvalues
    if (threadIdx.x < c.width)
    {
        volatile jvalue &val = jvalues[threadIdx.x];
        auto j = threadIdx.x - c.jmax;
        if (j == -c.jmax)
        {
            val.pu = PU_C(j, c.M);
            val.pm = PM_C(j, c.M);
            val.pd = PD_C(j, c.M);
        }
        else if (j == c.jmax)
        {
            val.pu = PU_B(j, c.M);
            val.pm = PM_B(j, c.M);
            val.pd = PD_B(j, c.M);
        }
        else
        {
            val.pu = PU_A(j, c.M);
            val.pm = PM_A(j, c.M);
            val.pd = PD_A(j, c.M);
        }
    }

    // Forward induction to calculate Qs and alphas
    if (threadIdx.x == c.jmax)
    {
        Qs[c.jmax] = one;                  // Qs[0] = 1$
        alphas[0] = getYieldAtYear(c.dt);  // initial dt-period interest rate
    }
    else
    {
        Qs[threadIdx.x] = zero;
        QsCopy[threadIdx.x] = zero;
    }
    __syncthreads();

    for (auto i = 0; i < 1; ++i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = alphas[i];

        // Forward iteration step, compute Qs in the next time step
        auto j = threadIdx.x - jhigh;
        if (j <= jhigh)
        {
            auto jind = threadIdx.x + c.jmax; // array index for j
            auto &jval = jvalues[jind];      // precomputed probabilities
            auto qexp = Qs[jind] * exp(-(alpha + j * c.dr) * c.dt);

            if (j == -c.jmax)
            {
                // Bottom edge branching
                atomicAdd((real*)QsCopy + jind + 2, jval.pu * qexp); // up two
                atomicAdd((real*)QsCopy + jind + 1, jval.pm * qexp); // up one
                atomicAdd((real*)QsCopy + jind, jval.pd * qexp);     // middle
            }
            else if (j == c.jmax)
            {
                // Top edge branching
                atomicAdd((real*)QsCopy + jind, jval.pu * qexp);     // middle
                atomicAdd((real*)QsCopy + jind - 1, jval.pm * qexp); // down one
                atomicAdd((real*)QsCopy + jind - 2, jval.pd * qexp); // down two
            }
            else
            {
                // Standard branching
                printf("standard\n");
                atomicAdd((real*)(QsCopy + jind + 1), jval.pu * qexp); // up
                atomicAdd((real*)(QsCopy + jind), jval.pm * qexp);     // middle
                atomicAdd((real*)(QsCopy + jind - 1), jval.pd * qexp); // down
            }
        }
        __syncthreads();

        // Determine the new alpha using equation 30.22
        // by summing up Qs from the next time step
        auto jhigh1 = min(i + 1, c.jmax) + c.jmax;
        if (threadIdx.x < jhigh1)
        {
            auto j = threadIdx.x - c.jmax;
            QsCopy[threadIdx.x] *= exp(-j * c.dr * c.dt);
            printf("%d %f %d\n", j, QsCopy[threadIdx.x], blockIdx.x);
        }
        __syncthreads();

        QsCopy[threadIdx.x] = scanIncBlock<Add<int>>(QsCopy, threadIdx.x);
        __syncthreads();

        if (threadIdx.x == 0)
        {
            alphas[i + 1] = computeAlpha(QsCopy[0], i, c.dt);
            printf("%f %d\n", alphas[i + 1], blockIdx.x);
        }
        __syncthreads();

        // Switch Qs
        auto QsT = Qs;
        Qs = QsCopy;
        QsCopy = QsT;

        QsCopy[threadIdx.x] = 0;
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        printf("%d %d\n", c.n, blockIdx.x);
        res[blockIdx.x] = c.n;
    }
    __syncthreads();
}

void computeCuda(OptionConstants *options, real *result, int count, int n_max, int width_max, bool isTest = false)
{
    // Maximum width has to fit into a block that should be a multiple of 32.
    int width_rem = width_max % 32;
    int blockSize = width_rem == 0 ? width_max : (width_max + 32 - width_rem);

    const unsigned int shMemSize = (width_max * 2 + n_max + 1) * sizeof(real) + width_max * sizeof(jvalue);

    if (isTest)
    {
        cout << "Running trinomial option pricing for " << count << " options with block size " << blockSize << endl;
        cout << "Shared memory size: " << shMemSize << endl;
    }

    auto time_begin = steady_clock::now();

    real *d_result;
    OptionConstants *d_options;
    CudaSafeCall(cudaMalloc((void **)&d_result, count * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_options, count * sizeof(OptionConstants)));

    cudaMemcpy(d_options, options, count * sizeof(OptionConstants), cudaMemcpyHostToDevice);

    auto time_begin_kernel = steady_clock::now();
    computeSingleOptionKernel<<<count, blockSize, shMemSize>>>(d_result, d_options, n_max);
    cudaThreadSynchronize();
    auto time_end_kernel = steady_clock::now();

    CudaCheckError();

    // Copy result
    cudaMemcpy(result, d_result, count * sizeof(real), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
    cudaFree(d_options);

    auto time_end = steady_clock::now();
    if (isTest)
    {
        cout << "Kernel executed in " << duration_cast<milliseconds>(time_end_kernel - time_begin_kernel).count() << " ms" << endl;
        cout << "Total GPU time: " << duration_cast<milliseconds>(time_end - time_begin).count() << " ms" << endl
             << endl;
    }
}


void computeAllOptions(const string &filename, bool isTest = false)
{
    // Read options from filename, allocate the result array
    auto options = Option::read_options(filename);
    auto result = new real[options.size()];
    auto optionConstants = new OptionConstants[options.size()];
    int width_max = 0;
    int n_max = 0;

    for (int i = 0; i < options.size(); ++i)
    {
        auto c = OptionConstants::computeConstants(options.at(i));
        optionConstants[i] = c;
        if (c.n > n_max) n_max = c.n;
        if (c.width > width_max) width_max = c.width;
    }

    computeCuda(optionConstants, result, options.size(), n_max, width_max, isTest);

    FutharkArrays::write_futhark_array(result, options.size());

    delete[] result;
}

int main(int argc, char *argv[])
{
    bool isTest = false;
    string filename;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-test") == 0)
        {
            isTest = true;
        }
        else
        {
            filename = argv[i];
        }
    }

    computeAllOptions(filename, isTest);

    return 0;
}
