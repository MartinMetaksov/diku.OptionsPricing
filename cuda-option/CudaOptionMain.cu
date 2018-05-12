
// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#include "../common/Args.hpp"
#include "Version1.cuh"
#include "Version2.cuh"
#include "Version3.cuh"

using namespace std;
using namespace trinom;

void run(const Options &options, const Yield &yield, vector<real> &results, const Args &args)
{
    auto time_begin = steady_clock::now();

    switch (args.version)
    {
        case 1:
            cuda::computeOptionsNaive(options, yield, results, 64, args.sort, args.test);
            break;
        case 2:
            cuda::computeOptionsCoalesced(options, yield, results, 64, args.sort, args.test);
            break;
        case 3:
            cuda::computeOptionsWithPaddingPerThreadBlock(options, yield, results, 64, args.sort, args.test);
            break;
    }

    auto time_end = steady_clock::now();
    
    if (args.test)
    {
        cout << "Total execution time " << duration_cast<microseconds>(time_end - time_begin).count() << " microsec" << endl;
    }
}

void computeAllOptions(const Args &args)
{
    if (args.test)
    {
        cout << "Cuda one option per thread version " << args.version << endl;
    }
    
    // Read options and yield curve.
    Options options(args.options);
    Yield yield(args.yield);

    cudaFree(0);

    if (args.test && args.runs > 0)
    {
        cout << "Performing " << args.runs << " runs" << endl;
        for (auto i = 0; i < args.runs; ++i)
        {
            cout << "----------------" << endl;
            vector<real> results;
            results.resize(options.N);
            run(options, yield, results, args);
            cout << "----------------" << endl;
        }
    }

    vector<real> results;
    results.resize(options.N);
    run(options, yield, results, args);
    
    if (!args.test)
    {
        Arrays::write_array(cout, results);
    }
}

int main(int argc, char *argv[])
{
    Args args(argc, argv);

    computeAllOptions(args);

    return 0;
}
