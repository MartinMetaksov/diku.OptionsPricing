
// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#include "Version1.cuh"
#include "Version2.cuh"
#include "Version3.cuh"

using namespace std;
using namespace trinom;

void computeAllOptions(const Args &args)
{
    if (args.test)
    {
        cout << "Cuda one option per thread version " << args.version << endl;
    }

    // Read options from filename, allocate the result array
    auto options = Option::readOptions(args.options);
    auto yield = Yield::readYieldCurve(args.yield);

    cudaFree(0);
    auto time_begin = steady_clock::now();

    vector<OptionConstants> optionConstants;
    optionConstants.reserve(options.size());
    for (auto &option : options)
    {
        auto constant = OptionConstants::computeConstants(option);
        optionConstants.push_back(constant);
    }

    OptionConstants::sortConstants(optionConstants, args.sort, args.test);
    CudaSafeCall(cudaMemcpyToSymbol(cuda::YieldCurve, yield.data(), yield.size() * sizeof(Yield)));

    vector<real> results;
    results.resize(optionConstants.size());

    switch (args.version)
    {
        case 1:
            cuda::computeOptionsNaive(optionConstants, yield.size(), results, args.test);
            break;
        case 2:
            cuda::computeOptionsCoalesced(optionConstants, yield.size(), results, args.test);
            break;
        case 3:
            cuda::computeOptionsWithPaddingPerThreadBlock(optionConstants, yield.size(), results, args.test);
            break;
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
    auto args = Args::parseArgs(argc, argv);

    computeAllOptions(args);

    return 0;
}
