
// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#include "../common/Args.hpp"
#include "Version1.cuh"
// #include "Version2.cuh"
// #include "Version3.cuh"

using namespace std;
using namespace trinom;

void computeAllOptions(const Args &args)
{
    if (args.test)
    {
        cout << "Cuda one option per thread version " << args.version << endl;
    }
    
    // Read options and yield curve, allocate the result array.
    auto options = Options(args.options);
    auto yieldSize = cuda::readYieldCurve(args.yield);

    auto time_begin = steady_clock::now();

    vector<real> results;
    results.resize(options.N);

    switch (args.version)
    {
        case 1:
            cuda::computeOptionsNaive(options, yieldSize, results, 64, args.test);
            break;
        // case 2:
        //     cuda::computeOptionsCoalesced(options, yieldSize, results, args.test);
        //     break;
        // case 3:
        //     cuda::computeOptionsWithPaddingPerThreadBlock(options, yieldSize, results, args.test);
            // break;
    }

    auto time_end = steady_clock::now();

    if (!args.test)
    {
        Arrays::write_array(cout, results);
    }
    else
    {
        cout << "Total execution time " << duration_cast<milliseconds>(time_end - time_begin).count() << " ms" << endl;
    }
}

int main(int argc, char *argv[])
{
    Args args(argc, argv);

    computeAllOptions(args);

    return 0;
}
