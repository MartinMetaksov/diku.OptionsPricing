#ifndef CUDA_VERSION_3_CUH
#define CUDA_VERSION_3_CUH

#include "../cuda/CudaDomain.cuh"

using namespace chrono;
using namespace trinom;

namespace cuda
{

__device__ inline real* getUnpaddedArrayAt(const int index, real *array, const int threadId, const int blockSize, const int blockStart)
{
    return array + blockStart + blockSize * index + threadId;
}

__device__ void fillUnpaddedArrayColumn(const int count, const real value, real *array, const int threadId, const int blockSize, const int blockStart)
{
    auto ptr = getUnpaddedArrayAt(0, array, threadId, blockSize, blockStart);
    for (auto i = 0; i < count; ++i)
    {
        *ptr = value;
        ptr += blockSize;
    }
}

__global__ void
kernelPaddingPerThreadBlock(real *res, const OptionConstants *options, real *QsAll, real *QsCopyAll, real *alphasAll, int *ScannedWidths, int *ScannedHeights, const int totalCount, const int yieldCurveSize)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // const int blockSize = ;
    const int widthBlockStartIndex = ScannedWidths[blockIdx.x];
    const int heightBlockStartIndex = ScannedHeights[blockIdx.x];
    // Out of options check
    if (idx >= totalCount) return;

    auto c = options[idx];
    auto alpha = getYieldAtYear(c.dt, c.termUnit, YieldCurve, yieldCurveSize);
    *getUnpaddedArrayAt(c.jmax, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) = one;
    *getUnpaddedArrayAt(0, alphasAll, threadIdx.x, blockDim.x, heightBlockStartIndex) = alpha;

    for (auto i = 1; i <= c.n; ++i)
    {
        const auto jhigh = min(i, c.jmax);
        real alpha_val = 0;

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - (-c.jmax); // array index for j
            
            auto expp1 = j == jhigh ? zero : *getUnpaddedArrayAt(jind + 1, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) * exp(-(alpha + computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);
            auto expm = *getUnpaddedArrayAt(jind, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) * exp(-(alpha + computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);
            auto expm1 = j == -jhigh ? zero : *getUnpaddedArrayAt(jind - 1, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex)  * exp(-(alpha + computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);
            real Q;

            if (i == 1) {
                if (j == -jhigh) {
                    Q = computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                } else if (j == jhigh) {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1;
                } else {
                    Q = computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm;
                }
            }
            else if (i <= c.jmax) {
                if (j == -jhigh) {
                    Q = computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                } else if (j == -jhigh + 1) {
                    Q = computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                } else if (j == jhigh) {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1;
                } else if (j == jhigh - 1) {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm;
                } else {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                }
            } else {
                if (j == -jhigh) {
                    Q = computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                } else if (j == -jhigh + 1) {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 2) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1;
                            
                } else if (j == jhigh) {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * expm;
                } else if (j == jhigh - 1) {
                    Q = computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 2) * expp1;
                            
                } else {
                    Q = ((j == -jhigh + 2) ? computeJValue(jind - 2, c.dr, c.M, c.width, c.jmax, 1) * *getUnpaddedArrayAt(jind - 2, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) * exp(-(alpha + computeJValue(jind - 2, c.dr, c.M, c.width, c.jmax, 0)) * c.dt) : zero) +
                        computeJValue(jind - 1, c.dr, c.M, c.width, c.jmax, 1) * expm1 +
                        computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * expm +
                        computeJValue(jind + 1, c.dr, c.M, c.width, c.jmax, 3) * expp1 +
                        ((j == jhigh - 2) ? computeJValue(jind + 2, c.dr, c.M, c.width, c.jmax, 3) * *getUnpaddedArrayAt(jind + 2, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) * exp(-(alpha + computeJValue(jind + 2, c.dr, c.M, c.width, c.jmax, 0)) * c.dt) : zero);
                }
            }
            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            *getUnpaddedArrayAt(jind, QsCopyAll, threadIdx.x, blockDim.x, widthBlockStartIndex) = Q;
            alpha_val += Q * exp(-computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0) * c.dt);
        }

        alpha = computeAlpha(alpha_val, i-1, c.dt, c.termUnit, YieldCurve, yieldCurveSize);
        *getUnpaddedArrayAt(i, alphasAll, threadIdx.x, blockDim.x, heightBlockStartIndex) = alpha;

        // Switch Qs
        auto QsT = QsAll;
        QsAll = QsCopyAll;
        QsCopyAll = QsT;
        fillUnpaddedArrayColumn(c.width, 0, QsCopyAll, threadIdx.x, blockDim.x, widthBlockStartIndex);
    }
    
    // Backward propagation
    fillUnpaddedArrayColumn(c.width, 100, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex);


    for (auto i = c.n - 1; i >= 0; --i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = *getUnpaddedArrayAt(i, alphasAll, threadIdx.x, blockDim.x, heightBlockStartIndex);
        
        auto isMaturity = i == ((int)(c.t / c.dt));

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j
            auto callExp = exp(-(alpha + computeJValue(jind, c.dr, c.M, c.width, c.jmax, 0)) * c.dt);

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * *getUnpaddedArrayAt(jind, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * *getUnpaddedArrayAt(jind - 1, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * *getUnpaddedArrayAt(jind - 2, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex)) *
                      callExp;
            }
            else if (j == -c.jmax)
            {
                // Bottom edge branching
                res = (computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * *getUnpaddedArrayAt(jind + 2, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * *getUnpaddedArrayAt(jind + 1, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * *getUnpaddedArrayAt(jind, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex)) *
                      callExp;
            }
            else
            {
                // Standard branching
                res = (computeJValue(jind, c.dr, c.M, c.width, c.jmax, 1) * *getUnpaddedArrayAt(jind + 1, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 2) * *getUnpaddedArrayAt(jind, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex) +
                    computeJValue(jind, c.dr, c.M, c.width, c.jmax, 3) * *getUnpaddedArrayAt(jind - 1, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex)) *
                      callExp;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            *getUnpaddedArrayAt(jind, QsCopyAll, threadIdx.x, blockDim.x, widthBlockStartIndex) = computeCallValue(isMaturity, c, res);
        }

        // Switch call arrays
        auto QsT = QsAll;
        QsAll = QsCopyAll;
        QsCopyAll = QsT;

        fillUnpaddedArrayColumn(c.width, 0, QsCopyAll, threadIdx.x, blockDim.x, widthBlockStartIndex);
    }

    res[idx] = *getUnpaddedArrayAt(c.jmax, QsAll, threadIdx.x, blockDim.x, widthBlockStartIndex);
}

void computeOptionsWithPaddingPerThreadBlock(const vector<OptionConstants> &options, const int yieldSize, vector<real> &results, bool isTest = false)
{
    const auto count = options.size();
    const auto blockSize = 64;
    const auto blockCount = ceil(count / ((float)blockSize));

    // Compute padding
    vector<int> maxWidths(blockCount, 0);
    vector<int> maxHeights(blockCount, 0);
    int blockCounter = 0;
    
    for (int i = 0; i < options.size(); ++i) {
        auto &option = options.at(i);
        if (option.width > maxWidths.at(blockCounter))
        {
            maxWidths.at(blockCounter) = option.width;
        }
        if (option.n >= maxHeights.at(blockCounter))
        {
            maxHeights.at(blockCounter) = option.n + 1; // n + 1 alphas
        }   

        if ((i+1) % blockSize == 0) {
            blockCounter++;
        }     
    }

    int totalQsCount = 0;
    int totalAlphasCount = 0;

    // todo: this can maybe be done better in c++ ? :D 
    vector<int> scannedWidths(blockCount, 0);
    vector<int> scannedHeights(blockCount, 0);
    if (maxWidths.size() == 1) {
        totalQsCount += maxWidths.at(0) * blockSize;
    }
    for (int i = 1; i < maxWidths.size(); i++) {
        auto& n = maxWidths.at(i-1);
        totalQsCount += n * blockSize;
        scannedWidths.at(i) = scannedWidths.at(i-1) + n * blockSize;
        if (i == maxWidths.size()-1) {
            totalQsCount += maxWidths.at(i) * blockSize;
        }
    }

    if (maxHeights.size() == 1) {
        totalAlphasCount += maxHeights.at(0) * blockSize;
    }
    for (int i = 1; i < maxHeights.size(); i++) {
        auto& n = maxHeights.at(i-1);
        totalAlphasCount += n * blockSize;
        scannedHeights.at(i) = scannedHeights.at(i-1) + n * blockSize;
        if (i == maxHeights.size()-1) {
            totalAlphasCount += maxHeights.at(i) * blockSize;
        }
    }

    if (isTest)
    {
        int memorySize = count * sizeof(real) + count * sizeof(OptionConstants)
                        + 2 * totalQsCount * sizeof(real) + totalAlphasCount * sizeof(real);
        cout << "Running trinomial option pricing for " << count << " options with block size " << blockSize << endl;
        cout << "Global memory size " << memorySize / (1024.0 * 1024.0) << " MB" << endl;
    }

    const auto time_begin = steady_clock::now();

    real *d_result, *d_Qs, *d_QsCopy, *d_alphas;
    int *d_ScannedWidths, *d_ScannedHeights;
    OptionConstants *d_options;
    CudaSafeCall(cudaMalloc((void **)&d_result, count * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_options, count * sizeof(OptionConstants)));
    CudaSafeCall(cudaMalloc((void **)&d_Qs, totalQsCount * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_QsCopy, totalQsCount * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_ScannedWidths, blockCount * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_ScannedHeights, blockCount * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_alphas, totalAlphasCount * sizeof(real)));

    CudaSafeCall(cudaMemcpy(d_options, options.data(), count * sizeof(OptionConstants), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_ScannedWidths, scannedWidths.data(), blockCount * sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_ScannedHeights, scannedHeights.data(), blockCount * sizeof(int), cudaMemcpyHostToDevice));

    auto time_begin_kernel = steady_clock::now();
    kernelPaddingPerThreadBlock<<<blockCount, blockSize>>>(d_result, d_options, d_Qs, d_QsCopy, d_alphas, d_ScannedWidths, d_ScannedHeights, count, yieldSize);
    cudaThreadSynchronize();
    auto time_end_kernel = steady_clock::now();

    CudaCheckError();

    // Copy result
    cudaMemcpy(results.data(), d_result, count * sizeof(real), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
    cudaFree(d_options);
    cudaFree(d_Qs);
    cudaFree(d_QsCopy);
    cudaFree(d_alphas);
    cudaFree(d_ScannedWidths);
    cudaFree(d_ScannedHeights);

    auto time_end = steady_clock::now();
    if (isTest)
    {
        cout << "Kernel executed in " << duration_cast<milliseconds>(time_end_kernel - time_begin_kernel).count() << " ms" << endl;
        cout << "Total GPU time: " << duration_cast<milliseconds>(time_end - time_begin).count() << " ms" << endl
            << endl;
    }
}
}

#endif
