#include "Seq.hpp"
#include "../common/Arrays.hpp"
#include "../common/Args.hpp"
#include <chrono>

using namespace trinom;
using namespace chrono;

void computeAllOptions(const Args &args)
{
    // Read options from filename, allocate the result array
    Options options(args.options);
    Yield yield(args.yield);

    auto time_begin = steady_clock::now();

    vector<real> results;
    results.reserve(options.N);

    for (auto i = 0; i < options.N; ++i)
    {
        OptionConstants c(options, i);
        auto result = seq::computeSingleOption(c, yield);
        results.push_back(result);
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
