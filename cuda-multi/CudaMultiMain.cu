
// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#include "../common/Args.hpp"
#include "Version1.cuh"
#include "Version2.cuh"

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
    Args args(argc, argv);

    computeAllOptions(args);

    return 0;
}
