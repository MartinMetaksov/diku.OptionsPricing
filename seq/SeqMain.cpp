#include "../common/Domain.hpp"
#include "../common/OptionConstants.hpp"
#include "../common/Arrays.hpp"
#include "../common/Args.hpp"
#include "Seq.hpp"

using namespace trinom;

void computeAllOptions(const Args &args)
{
    // Read options from filename, allocate the result array
    auto options = Option::readOptions(args.options);
    auto yield = Yield::readYieldCurve(args.yield);
    auto result = new real[options.size()];

    for (int i = 0; i < options.size(); ++i)
    {
        auto c = OptionConstants::computeConstants(options.at(i));
        result[i] = seq::computeSingleOption(c, yield);
    }

    Arrays::write_array(result, options.size());

    delete[] result;
}

int main(int argc, char *argv[])
{
    auto args = Args::parseArgs(argc, argv);

    computeAllOptions(args);

    return 0;
}
