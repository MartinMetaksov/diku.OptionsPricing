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

__global__ void
computeSingleOptionKernel(real *res, OptionConstants *options, int n_max)
{
    extern __shared__ real sh_mem[];

    volatile real *Qs = (real *)&sh_mem;
    volatile real *QsCopy = &Qs[blockDim.x];
    volatile real *alphas = &QsCopy[blockDim.x];

    auto option = options[blockIdx.x];

    res[blockIdx.x] = option.n;
}

void computeCuda(OptionConstants *options, real *result, int count, int n_max, int width_max, bool isTest = false)
{
    // Maximum width has to fit into a block that should be a multiple of 32.
    int width_rem = width_max % 32;
    int blockSize = width_rem == 0 ? width_max : (width_max + 32 - width_rem);

    const unsigned int shMemSize = (width_max * 2 + n_max + 1) * sizeof(real);

    if (isTest)
    {
        cout << "Running trinomial option pricing with block size " << blockSize << endl;
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
