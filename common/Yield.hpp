#ifndef YIELD_HPP
#define YIELD_HPP

#include <stdexcept>
#include <fstream>
#include <vector>
#include "Real.hpp"
#include "Arrays.hpp"

using namespace std;

namespace trinom
{

struct Yield
{
    int N;
    vector<real> Prices;
    vector<int32_t> TimeSteps;

    Yield(const string &filename)
    {
        if (filename.empty())
        {
            throw invalid_argument("File not specified.");
        }

        ifstream in(filename);

        if (!in)
        {
            throw invalid_argument("File does not exist.");
        }

        Arrays::read_array(in, Prices);
        Arrays::read_array(in, TimeSteps);
        N = Prices.size();

        in.close();
    }
};
}

#endif
