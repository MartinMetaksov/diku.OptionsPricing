#ifndef OPTION_HPP
#define OPTION_HPP

#include <fstream>
#include <vector>
#include "Real.hpp"
#include "FutharkArrays.hpp"

using namespace std;

struct Option
{
  public:
    real StrikePrice;
    real Maturity;
    real Length;
    int TermUnit;
    int TermStepCount;
    real ReversionRate;
    real Volatility;

    static vector<Option> read_options(const string &filename)
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

        vector<real> strikes;
        vector<real> maturities;
        vector<real> lengths;
        vector<int> termunits;
        vector<int> termsteps;
        vector<real> rrps;
        vector<real> vols;

        FutharkArrays::read_futhark_array(in, &strikes);
        FutharkArrays::read_futhark_array(in, &maturities);
        FutharkArrays::read_futhark_array(in, &lengths);
        FutharkArrays::read_futhark_array(in, &termunits);
        FutharkArrays::read_futhark_array(in, &termsteps);
        FutharkArrays::read_futhark_array(in, &rrps);
        FutharkArrays::read_futhark_array(in, &vols);

        int size = strikes.size();
        vector<Option> options;
        options.reserve(size);

        for (int i = 0; i < size; ++i)
        {
            Option o;
            o.StrikePrice = strikes.at(i);
            o.Maturity = maturities.at(i);
            o.Length = lengths.at(i);
            o.TermUnit = termunits.at(i);
            o.TermStepCount = termsteps.at(i);
            o.ReversionRate = rrps.at(i);
            o.Volatility = vols.at(i);
            options.push_back(o);
        }
        return options;
    }
};

#endif