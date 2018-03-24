#define CUDA

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#include "../common/Real.hpp"
#include "../common/OptionConstants.hpp"
#include "../common/FutharkArrays.hpp"
#include "../common/Domain.hpp"
#include "../test/Mock.hpp"
#include "../cuda/CudaErrors.cuh"

#include <chrono>

using namespace std;
using namespace chrono;

__global__ void
computeSingleOptionKernel(real *res, OptionConstants *options, real *QsAll, real *QsCopyAll, real *alphasAll, int *QsInd, int *alphasInd, int totalCount)
{
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Out of options check
    if (idx >= totalCount) return;

    auto c = options[idx];
    auto Qs = QsAll + QsInd[idx];
    auto QsCopy = QsCopyAll + QsInd[idx];
    auto alphas = alphasAll + alphasInd[idx];

    // Stress test
    for (auto a = 0; a < c.n + 1; ++ a)
    {
        for (auto i = 1; i < c.width - 1; ++i)
        {
            Qs[i] = QsCopy[i-1] * 10 + QsCopy[i] * 5 + QsCopy[i + 1];
        }
        alphas[a] = Qs[0] + 3;
    }

    // some test result
    res[idx] = c.n;
}

void computeCuda(OptionConstants *options, real *result, int count, bool isTest = false)
{
    // Compute indices
    int* QsInd = new int[count];
    int* alphasInd = new int[count];
    QsInd[0] = 0;
    alphasInd[0] = 0;
    int totalQsCount = 0;
    int totalAlphasCount = 0;
    for (auto i = 0; i < count - 1; ++i)
    {
        auto &option = options[i];
        totalQsCount += option.width;
        totalAlphasCount += option.n + 1;
        QsInd[i + 1] = totalQsCount;
        alphasInd[i + 1] = totalAlphasCount;
    }
    totalQsCount += options[count - 1].width;
    totalAlphasCount += options[count - 1].n + 1;
    
    auto blockSize = 32;
    auto blockCount = count / (blockSize + 1) + 1;

    if (isTest)
    {
        int memorySize = count * sizeof(real) + count * sizeof(OptionConstants) + 2 * count * sizeof(int)
                        + 2 * totalQsCount * sizeof(real) + totalAlphasCount * sizeof(real);
        cout << "Running trinomial option pricing for " << count << " options with block size " << blockSize << endl;
        cout << "Global memory size " << memorySize / (1024.0 * 1024.0) << " MB" << endl;
    }

    auto time_begin = steady_clock::now();

    real *d_result, *d_Qs, *d_QsCopy, *d_alphas;
    int *d_QsInd, *d_alphasInd;
    OptionConstants *d_options;
    CudaSafeCall(cudaMalloc((void **)&d_result, count * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_options, count * sizeof(OptionConstants)));
    CudaSafeCall(cudaMalloc((void **)&d_QsInd, count * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_alphasInd, count * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_Qs, totalQsCount * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_QsCopy, totalQsCount * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_alphas, totalAlphasCount * sizeof(real)));

    cudaMemcpy(d_options, options, count * sizeof(OptionConstants), cudaMemcpyHostToDevice);
    cudaMemcpy(d_QsInd, QsInd, count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alphasInd, alphasInd, count * sizeof(int), cudaMemcpyHostToDevice);

    auto time_begin_kernel = steady_clock::now();
    computeSingleOptionKernel<<<blockCount, blockSize>>>(d_result, d_options, d_Qs, d_QsCopy, d_alphas, d_QsInd, d_alphasInd, count);
    cudaThreadSynchronize();
    auto time_end_kernel = steady_clock::now();

    CudaCheckError();

    // Copy result
    cudaMemcpy(result, d_result, count * sizeof(real), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
    cudaFree(d_options);
    cudaFree(d_QsInd);
    cudaFree(d_alphasInd);
    cudaFree(d_Qs);
    cudaFree(d_QsCopy);
    cudaFree(d_alphas);

    auto time_end = steady_clock::now();
    if (isTest)
    {
        cout << "Kernel executed in " << duration_cast<milliseconds>(time_end_kernel - time_begin_kernel).count() << " ms" << endl;
        cout << "Total GPU time: " << duration_cast<milliseconds>(time_end - time_begin).count() << " ms" << endl
             << endl;
    }
}


void computeAllOptions(const string &filename, bool isTest, int mockCount)
{
    OptionConstants* optionConstants;
    int length;

    if (mockCount > 0)
    {
        // Make mock constants, count should be a multiple of 4
        length = mockCount;
        optionConstants = new OptionConstants[length];
        for (auto i = 0; i < length; i += 4)
        {
            Mock::mockConstants(optionConstants + i + 1, 1, 101, 12000);
            Mock::mockConstants(optionConstants + i, 1, 10001, 1200);
            Mock::mockConstants(optionConstants + i + 2, 1, 11, 1800);
            Mock::mockConstants(optionConstants + i + 3, 1, 1001, 12);
        }
    }
    else
    {
        // Read options from filename, allocate the result array
        auto options = Option::read_options(filename);
        length = options.size();
        optionConstants = new OptionConstants[length];

        for (auto i = 0; i < length; ++i)
        {
            optionConstants[i] = OptionConstants::computeConstants(options.at(i));
        }
    }

    auto result = new real[length];
    computeCuda(optionConstants, result, length, isTest);

    // Don't print mock results
    if (mockCount <= 0)
    {
        FutharkArrays::write_futhark_array(result, length);
    }

    delete[] result;
    delete[] optionConstants;
}

int main(int argc, char *argv[])
{
    bool isTest = false;
    int mockCount = -1;
    string filename;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-test") == 0)
        {
            isTest = true;
        }
        else if (strcmp(argv[i], "-mock") == 0)
        {
            ++i;
            mockCount = stoi(argv[i]);
        }
        else
        {
            filename = argv[i];
        }
    }

    computeAllOptions(filename, isTest, mockCount);

    return 0;
}
