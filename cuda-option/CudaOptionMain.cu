
// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#include "CudaOption.cuh"
#include "../common/Arrays.hpp"
#include "../common/Args.hpp"

using namespace std;
using namespace trinom;

void computeAllOptions(const Args &args)
{
    // Read options from filename, allocate the result array
    auto options = Option::readOptions(args.options);
    auto yield = Yield::readYieldCurve(args.yield);
    vector<OptionConstants> optionConstants;
    optionConstants.reserve(options.size());

    for (auto &option : options)
    {
        auto constant = OptionConstants::computeConstants(option);
        optionConstants.push_back(constant);
    }

    if (args.test)
        cout << "Cuda option version " << args.version << endl;

    OptionConstants::sortConstants(optionConstants, args.sort, args.test);

    vector<real> results;
    results.resize(options.size());

    switch (args.version)
    {
        case 1:
            cuda::computeOptionsNaive(optionConstants, yield, results, args.test);
            break;
        case 2:
            cuda::computeOptionsCoalesced(optionConstants, yield, results, args.test);
            break;
        case 3:
            cuda::computeOptionsWithPaddingPerThreadBlock(optionConstants, yield, results, args.test);
            break;
    }

    if (!args.test)
    {
        Arrays::write_array(cout, results);
    }
}

int main(int argc, char *argv[])
{
    auto args = Args::parseArgs(argc, argv);

    computeAllOptions(args);

    return 0;
}
