
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
    auto length = options.size();
    auto optionConstants = new OptionConstants[length];

    for (auto i = 0; i < length; ++i)
    {
        optionConstants[i] = OptionConstants::computeConstants(options.at(i));
    }

    auto result = new real[length];
    cuda::computeOptions(optionConstants, result, length, yield, args.test);

    if (!args.test)
    {
        Arrays::write_array(result, length);
    }

    delete[] result;
    delete[] optionConstants;
}

int main(int argc, char *argv[])
{
    auto args = Args::parseArgs(argc, argv);

    computeAllOptions(args);

    return 0;
}
