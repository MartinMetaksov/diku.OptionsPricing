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
    vector<real> results;
    results.reserve(options.size());

    for (auto &option : options)
    {
        auto c = OptionConstants::computeConstants(option);
        results.push_back(seq::computeSingleOption(c, yield));
    }

    Arrays::write_array(cout, results);
}

int main(int argc, char *argv[])
{
    auto args = Args::parseArgs(argc, argv);

    computeAllOptions(args);

    return 0;
}
