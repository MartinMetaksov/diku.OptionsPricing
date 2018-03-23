#define CUDA
#include "../common/Real.hpp"
#include "../common/OptionConstants.hpp"
#include "../common/FutharkArrays.hpp"
#include "../common/Domain.hpp"
#include "../cuda/CudaErrors.cuh"

#include <chrono>

using namespace std;
using namespace chrono;

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

__global__ void
computeSingleOptionKernel(real *res, OptionConstants *options, real *QsAll, real *QsCopyAll, real *alphasAll, int *QsInd, int *alphasInd)
{
    auto c = options[threadIdx.x];
    auto Qs = QsAll + QsInd[threadIdx.x];
    auto QsCopy = QsCopyAll + QsInd[threadIdx.x];
    auto alphas = alphasAll + alphasInd[threadIdx.x];

    // some test result
    res[threadIdx.x] = c.n;
}

void computeCuda(OptionConstants *options, real *result, int count, bool isTest = false)
{
    auto blockSize = 1;
    
    if (isTest)
    {
        cout << "Running trinomial option pricing for " << count << " options with block size " << blockSize << endl;
    }

    auto time_begin = steady_clock::now();

    real *d_result, *d_Qs, *d_QsCopy, *d_alphas;
    int *d_QsInd, *d_alphasInd;
    OptionConstants *d_options;
    CudaSafeCall(cudaMalloc((void **)&d_result, count * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_options, count * sizeof(OptionConstants)));

    cudaMemcpy(d_options, options, count * sizeof(OptionConstants), cudaMemcpyHostToDevice);

    auto time_begin_kernel = steady_clock::now();
    computeSingleOptionKernel<<<count, blockSize>>>(d_result, d_options, d_Qs, d_QsCopy, d_alphas, d_QsInd, d_alphasInd);
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

    for (int i = 0; i < options.size(); ++i)
    {
        optionConstants[i] = OptionConstants::computeConstants(options.at(i));
    }

    computeCuda(optionConstants, result, options.size(), isTest);

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
