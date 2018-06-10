#include <chrono>

#include "../common/Args.hpp"
#include "../common/Arrays.hpp"
#include "Seq.hpp"

using namespace std;
using namespace chrono;
using namespace trinom;

void computeAllOptions(const Args &args)
{
    // Read options from filename, allocate the result array
    Options options(args.options);
    Yield yield(args.yield);

    auto time_begin = steady_clock::now();

    vector<real> results;
    results.reserve(options.N);

    seq::computeOptions(options, yield, results);

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
