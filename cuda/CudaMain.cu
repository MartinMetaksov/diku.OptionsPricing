#define CUDA
#include "../common/Real.hpp"
#include "../common/Option.hpp"
#include "../common/FutharkArrays.hpp"
#include "../common/Domain.hpp"
#include "CudaErrors.cuh"
#include "ScanKernels.cuh"
#include "FormattedOptions.hpp"

#include <chrono>

using namespace std;
using namespace chrono;

#define BEST_NUM_BLOCKS 256

#define SHARED_ARRAYS_BLOCK_REAL 2
#define SHARED_ARRAYS_BLOCK_INT 3
#define SHARED_ARRAYS_CHUNK_REAL 5
#define SHARED_ARRAYS_CHUNK_INT 6

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

__device__ void populate_flag_array(int *optionsInChunks, int maxOptionsInChunk, volatile int *flags, volatile int *scanned_lens, volatile int *ms)
{
    flags[threadIdx.x] = 0;
    __syncthreads();

    if (threadIdx.x < optionsInChunks[blockIdx.x])
    {
        if (threadIdx.x == 0)
        {
            flags[0] = scanned_lens[0];
        }
        else
        {
            flags[scanned_lens[threadIdx.x - 1]] = 2 * ms[threadIdx.x] + 1;
        }
    }
    else if (threadIdx.x < maxOptionsInChunk)
    {
        flags[scanned_lens[threadIdx.x - 1]] = blockDim.x - scanned_lens[optionsInChunks[blockIdx.x] - 1];
    }
}

__global__ void
trinomialChunk(
    real *res,
    Option *options,
    int *optionsInChunks,
    int *optionIndices,
    real *alphass,
    int maxOptionsInChunk,
    int ycCount,
    int n_max)
{
    extern __shared__ real sh_mem[];

    volatile real *Qs = (real *)&sh_mem;
    volatile real *QCopys = &Qs[blockDim.x];
    volatile int *iota2mp1 = (int *)&QCopys[blockDim.x];
    volatile int *flags = &iota2mp1[blockDim.x];
    volatile int *sgm_inds = &flags[blockDim.x];
    volatile real *Xs = (real *)&sgm_inds[blockDim.x];
    volatile real *dts = &Xs[maxOptionsInChunk];
    volatile real *drs = &dts[maxOptionsInChunk];
    volatile real *Ms = &drs[maxOptionsInChunk];
    volatile real *alpha_vals = &Ms[maxOptionsInChunk];
    volatile int *ns = (int *)&alpha_vals[maxOptionsInChunk];
    volatile int *imaxs = &ns[maxOptionsInChunk];
    volatile int *jmaxs = &imaxs[maxOptionsInChunk];
    volatile int *ms = &jmaxs[maxOptionsInChunk];
    volatile int *is = &ms[maxOptionsInChunk];
    volatile int *scanned_lens = &is[maxOptionsInChunk];
    real *block_alphass = alphass + ((n_max + 1) * maxOptionsInChunk * blockIdx.x);

    if (threadIdx.x < optionsInChunks[blockIdx.x])
    {
        Option option = options[optionIndices[blockIdx.x * maxOptionsInChunk + threadIdx.x]]; // actual option (within all blocks)
        Xs[threadIdx.x] = option.strike_price;
        ns[threadIdx.x] = option.num_of_terms;
        dts[threadIdx.x] = option.maturity / (real)option.num_of_terms;
        drs[threadIdx.x] = sqrt(three * (option.volatility * option.volatility * (one - exp(-two * option.reversion_rate * dts[threadIdx.x])) //
                                         / (two * option.reversion_rate)));
        Ms[threadIdx.x] = exp(-option.reversion_rate * dts[threadIdx.x]) - one;
        jmaxs[threadIdx.x] = (int)(minus184 / Ms[threadIdx.x]) + 1;
        ms[threadIdx.x] = jmaxs[threadIdx.x] + 2;
    }
    else if (threadIdx.x < maxOptionsInChunk)
    {
        Xs[threadIdx.x] = one;
        ns[threadIdx.x] = 0;
        dts[threadIdx.x] = one;
        drs[threadIdx.x] = one;
        Ms[threadIdx.x] = one;
        jmaxs[threadIdx.x] = -1;
        ms[threadIdx.x] = -1;
    }

    if (threadIdx.x < optionsInChunks[blockIdx.x])
    {
        sgm_inds[threadIdx.x] = 2 * ms[threadIdx.x] + 1;
    }
    else
    {
        sgm_inds[threadIdx.x] = 0;
    }
    __syncthreads();

    sgm_inds[threadIdx.x] = scanIncBlock<Add<int>>(sgm_inds, threadIdx.x);
    __syncthreads();

    if (threadIdx.x < maxOptionsInChunk)
    {
        scanned_lens[threadIdx.x] = sgm_inds[threadIdx.x];
    }
    __syncthreads();

    sgm_inds[threadIdx.x] = 0;
    __syncthreads();

    if (threadIdx.x > 0 && threadIdx.x < optionsInChunks[blockIdx.x])
    {
        sgm_inds[scanned_lens[threadIdx.x - 1]] = threadIdx.x;
    }
    else if (threadIdx.x >= optionsInChunks[blockIdx.x] && threadIdx.x < maxOptionsInChunk)
    {
        sgm_inds[scanned_lens[optionsInChunks[blockIdx.x] - 1]] = threadIdx.x;
    }

    populate_flag_array(optionsInChunks, maxOptionsInChunk, flags, scanned_lens, ms);
    __syncthreads();

    sgm_inds[threadIdx.x] = sgmScanIncBlock<Add<int>>(sgm_inds, flags, threadIdx.x);
    __syncthreads();

    iota2mp1[threadIdx.x] = 1;
    populate_flag_array(optionsInChunks, maxOptionsInChunk, flags, scanned_lens, ms);
    __syncthreads();

    iota2mp1[threadIdx.x] = sgmScanIncBlock<Add<int>>(iota2mp1, flags, threadIdx.x) - 1;
    __syncthreads();

    Qs[threadIdx.x] = (iota2mp1[threadIdx.x] == ms[sgm_inds[threadIdx.x]]) ? one : zero;

    if (threadIdx.x < maxOptionsInChunk)
    {
        block_alphass[threadIdx.x * (n_max + 1)] = h_YieldCurve[0].p;
    }
    __syncthreads();

    int sgm_ind = sgm_inds[threadIdx.x];
    int n = ns[sgm_ind];

    /*
    * Time stepping
    */
    for (int k = 0; k < n_max; ++k)
    {
        if (k < n)
        {
            if (threadIdx.x < maxOptionsInChunk)
            {
                imaxs[threadIdx.x] = min(k + 1, jmaxs[threadIdx.x]);
            }

            QCopys[threadIdx.x] = Qs[threadIdx.x];
        }
        __syncthreads();

        // forward iteration step
        if (k < n && sgm_ind < optionsInChunks[blockIdx.x])
        {
            int imax = imaxs[sgm_ind];
            int m = ms[sgm_ind];
            int j = iota2mp1[threadIdx.x] - m;
            if (j < (-imax) || j > imax)
            {
                Qs[threadIdx.x] = zero;
            }
            else
            {
                int begind = sgm_ind == 0 ? 0 : scanned_lens[sgm_ind - 1];
                Qs[threadIdx.x] = fwdHelper(threadIdx.x, Ms[sgm_ind], drs[sgm_ind], dts[sgm_ind], block_alphass[sgm_ind * (n_max + 1) + k], //
                                            QCopys, begind, m, k, imax, jmaxs[sgm_ind], j);
            }
        }
        __syncthreads();

        if (k < n)
        {
            int imax = imaxs[sgm_ind];
            int j = iota2mp1[threadIdx.x] - imax;
            if (j < (-imax) || j > imax || sgm_ind >= optionsInChunks[blockIdx.x])
            {
                QCopys[threadIdx.x] = zero;
            }
            else
            {
                int begind = sgm_ind == 0 ? 0 : scanned_lens[sgm_ind - 1];
                QCopys[threadIdx.x] = Qs[begind + j + ms[sgm_ind]] * exp(-((real)j) * drs[sgm_ind] * dts[sgm_ind]);
            }
        }

        populate_flag_array(optionsInChunks, maxOptionsInChunk, flags, scanned_lens, ms);
        __syncthreads();

        QCopys[threadIdx.x] = sgmScanIncBlock<Add<real>>(QCopys, flags, threadIdx.x);
        __syncthreads();

        if (threadIdx.x == blockDim.x - 1 || sgm_ind != sgm_inds[threadIdx.x + 1])
        {
            alpha_vals[sgm_ind] = QCopys[threadIdx.x];
        }
        __syncthreads();

        if (k < n && threadIdx.x < optionsInChunks[blockIdx.x])
        {
            real t = (k + 1) * dts[threadIdx.x] + one;
            int t2 = (int)((t >= zero ? 1 : -1) * half + abs(t));
            int t1 = t2 - 1;
            if (t2 >= ycCount)
            {
                t2 = ycCount - 1;
                t1 = ycCount - 2;
            }

            real R = (h_YieldCurve[t2].p - h_YieldCurve[t1].p) /
                         (h_YieldCurve[t2].t - h_YieldCurve[t1].t) *
                         (t * year - h_YieldCurve[t1].t) +
                     h_YieldCurve[t1].p;

            block_alphass[threadIdx.x * (n_max + 1) + (k + 1)] = log(alpha_vals[threadIdx.x] / exp(-R * t));
        }
        __syncthreads();
    }
    __syncthreads();

    volatile real *Calls = Qs;
    volatile real *CallCopys = QCopys;

    Calls[threadIdx.x] = ((iota2mp1[threadIdx.x] >= -jmaxs[sgm_ind] + ms[sgm_ind]) && (iota2mp1[threadIdx.x] <= jmaxs[sgm_ind] + ms[sgm_ind])) ? one : zero;
    __syncthreads();

    /*
     * Back propagation
     */
    for (int k = 0; k < n_max; ++k)
    {
        if (k < n)
        {
            if (threadIdx.x < optionsInChunks[blockIdx.x])
            {
                is[threadIdx.x] = ns[threadIdx.x] - 1 - k;
                imaxs[threadIdx.x] = min(is[threadIdx.x] + 1, jmaxs[threadIdx.x]);
            }
            else if (threadIdx.x < maxOptionsInChunk)
            {
                is[threadIdx.x] = 0;
                imaxs[threadIdx.x] = min(1, jmaxs[threadIdx.x]);
            }

            /*  
             * Copy array values to avoid overwriting during update 
             */
            CallCopys[threadIdx.x] = Calls[threadIdx.x];
        }
        __syncthreads();

        if (k < n && sgm_ind < optionsInChunks[blockIdx.x])
        {
            int imax = imaxs[sgm_ind];
            int isi = is[sgm_ind];
            int m = ms[sgm_ind];
            int j = iota2mp1[threadIdx.x] - m;
            if (j < (-imax) || j > imax)
            {
                Calls[threadIdx.x] = zero;
            }
            else
            {
                int begind = sgm_ind == 0 ? 0 : scanned_lens[sgm_ind - 1];
                Calls[threadIdx.x] = bkwdHelper(Xs[sgm_ind], Ms[sgm_ind], drs[sgm_ind], dts[sgm_ind], block_alphass[sgm_ind * (n_max + 1) + isi], //
                                                CallCopys, begind, m, isi, jmaxs[sgm_ind], j);
            }
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x < optionsInChunks[blockIdx.x])
    {
        int currentOptionIndex = optionIndices[blockIdx.x * maxOptionsInChunk + threadIdx.x]; // actual option index (within all blocks)
        int begind = threadIdx.x == 0 ? 0 : scanned_lens[threadIdx.x - 1];
        int m_ind = begind + ms[threadIdx.x];
        res[currentOptionIndex] = Calls[m_ind];
    }
}

void trinomialPricing(FormattedOptions format, real *result, bool isTest = false)
{
    const unsigned int sh_mem_size = SHARED_ARRAYS_CHUNK_REAL * format.max_options_in_chunk * sizeof(real) //
                                     + SHARED_ARRAYS_BLOCK_REAL * format.w * sizeof(real)                  //
                                     + SHARED_ARRAYS_CHUNK_INT * format.max_options_in_chunk * sizeof(int) //
                                     + SHARED_ARRAYS_BLOCK_INT * format.w * sizeof(int);

    if (isTest)
    {
        cout << "Running trinomial option pricing with block size " << format.w << ", " << format.max_options_in_chunk << " options in block" << endl;
        cout << "Shared memory size: " << sh_mem_size << endl;
    }
    auto time_begin = steady_clock::now();

    int ycCount = extent<decltype(h_YieldCurve)>::value;
    unsigned int num_chunks = format.options_in_chunk.size();

    real *d_result;
    Option *d_options;
    int *d_optionsInChunks;
    int *d_optionIndices;
    real *d_alphass;
    CudaSafeCall(cudaMalloc((void **)&d_result, format.options.size() * sizeof(real)));
    CudaSafeCall(cudaMalloc((void **)&d_options, format.options.size() * sizeof(Option)));
    CudaSafeCall(cudaMalloc((void **)&d_optionsInChunks, format.options_in_chunk.size() * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_optionIndices, format.option_indices.size() * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **)&d_alphass, format.option_indices.size() * (format.n_max + 1) * format.max_options_in_chunk * sizeof(real)));

    cudaMemcpy(d_options, format.options.data(), format.options.size() * sizeof(Option), cudaMemcpyHostToDevice);
    cudaMemcpy(d_optionsInChunks, format.options_in_chunk.data(), format.options_in_chunk.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_optionIndices, format.option_indices.data(), format.option_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    auto time_begin_kernel = steady_clock::now();
    trinomialChunk<<<num_chunks, format.w, sh_mem_size>>>(d_result, d_options, d_optionsInChunks, d_optionIndices, d_alphass, //
                                                          format.max_options_in_chunk, ycCount, format.n_max);
    cudaThreadSynchronize();
    auto time_end_kernel = steady_clock::now();

    CudaCheckError();

    // Copy result
    cudaMemcpy(result, d_result, format.options.size() * sizeof(real), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
    cudaFree(d_options);
    cudaFree(d_optionsInChunks);
    cudaFree(d_optionIndices);
    cudaFree(d_alphass);

    auto time_end = steady_clock::now();
    if (isTest)
    {
        cout << "Kernel executed in " << duration_cast<milliseconds>(time_end_kernel - time_begin_kernel).count() << " ms" << endl;
        cout << "Total GPU time: " << duration_cast<milliseconds>(time_end - time_begin).count() << " ms" << endl
             << endl;
    }
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

    // Read options from stdin, allocate the result array
    vector<Option> options = Option::read_options(filename);
    auto result = new real[options.size()];

    if (isTest)
    {
        cout << endl
             << "Initialized " << options.size() << " options of type "
             <<
#ifdef USE_DOUBLE
            "double"
#else
            "float"
#endif
             << endl
             << endl;

        for (int w = 128; w <= 512; w *= 2)
        {
            FormattedOptions format = FormattedOptions::format_options(w, options);

            trinomialPricing(format, result); // warmup
            trinomialPricing(format, result, true);
        }
    }
    else
    {
        // Compute the result
        FormattedOptions format = FormattedOptions::format_options(BEST_NUM_BLOCKS, options);
        trinomialPricing(format, result);

        // Print the result to stdin
        FutharkArrays::write_futhark_array(result, options.size());
    }

    return 0;
}
