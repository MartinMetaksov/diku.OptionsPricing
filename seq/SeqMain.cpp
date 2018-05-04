#include "Seq.hpp"
#include "../common/Arrays.hpp"
#include "../common/Args.hpp"

using namespace trinom;

void computeAllOptions(const Args &args)
{
    // Read options from filename, allocate the result array
    Options options(args.options);
    auto yield = Yield::readYieldCurve(args.yield);
    vector<real> results;
    results.reserve(options.N);

    for (auto i = 0; i < options.N; ++i)
    {
        OptionConstants c(options, i);
        auto result = seq::computeSingleOption(c, yield);
        results.push_back(result);
    }

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
