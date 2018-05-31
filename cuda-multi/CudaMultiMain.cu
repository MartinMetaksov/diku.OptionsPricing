
// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#include "Version1.cuh"
#include "Version2.cuh"
#include "Version3.cuh"
#include "../common/Args.hpp"

using namespace std;
using namespace trinom;

cuda::CudaRuntime run(const Options &options, const Yield &yield, vector<real> &results, const Args &args)
{
    switch (args.version)
    {
        case 1:
        {
            cuda::multi::KernelRunNaive kernelRun;
            kernelRun.run(options, yield, results, args.blockSize, args.sort, args.test);
            return kernelRun.runtime;
        }
        case 2:
        {
            cuda::multi::KernelRunCoalesced kernelRun;
            kernelRun.run(options, yield, results, args.blockSize, args.sort, args.test);
            return kernelRun.runtime;
        }
        case 3:
        {
            cuda::multi::KernelRunCoalescedBlock kernelRun;
            kernelRun.run(options, yield, results, args.blockSize, args.sort, args.test);
            return kernelRun.runtime;
        }
    }
    return cuda::CudaRuntime();
}

void computeAllOptions(const Args &args)
{
    if (args.test)
    {
        cout << "Cuda multiple options per thread block version " << args.version << endl;
    }
    
    // Read options and yield curve.
    Options options(args.options);
    Yield yield(args.yield);

    cudaFree(0);

    if (args.runs > 0)
    {
        cout << "Performing " << args.runs << " runs..." << endl;
        cuda::CudaRuntime best;
        for (auto i = 0; i < args.runs; ++i)
        {
            vector<real> results;
            results.resize(options.N);
            auto runtime = run(options, yield, results, args);
            if (runtime < best)
            {
                best = runtime;
            }
        }
        cout << "Best times: kernel " << best.KernelRuntime << " microsec, total " << best.TotalRuntime << " microsec." << endl;
    }
    else
    {
        vector<real> results;
        results.resize(options.N);
        run(options, yield, results, args);
        
        if (!args.test)
        {
            Arrays::write_array(cout, results);
        }
    }
}

int main(int argc, char *argv[])
{
    // Args args(argc, argv);
    Args args;
    args.options = "../data/options-60000.in";
    args.yield = "../data/yield.in";
    args.version = 3;
    args.blockSize = 256;
    args.runs = 0;
    args.test = false;
    args.sort = SortType::NONE;

    computeAllOptions(args);

    return 0;
}
