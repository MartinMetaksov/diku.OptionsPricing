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
    real p;
    int t;

    static vector<Yield> readYieldCurve(const string &filename)
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

        vector<real> prices;
        vector<int> timesteps;

        Arrays::read_array(in, prices);
        Arrays::read_array(in, timesteps);

        in.close();

        int size = prices.size();
        vector<Yield> curve;
        curve.reserve(size);

        for (int i = 0; i < size; ++i)
        {
            Yield y;
            y.p = prices.at(i);
            y.t = timesteps.at(i);
            curve.push_back(y);
        }
        return curve;
    }
};
}

#endif
