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
    real strike_price;
    real maturity;
    int num_of_terms;
    real reversion_rate;
    real volatility;

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
        vector<int> num_of_terms;
        vector<real> rrps;
        vector<real> vols;

        FutharkArrays::read_futhark_array(in, &strikes);
        FutharkArrays::read_futhark_array(in, &maturities);
        FutharkArrays::read_futhark_array(in, &num_of_terms);
        FutharkArrays::read_futhark_array(in, &rrps);
        FutharkArrays::read_futhark_array(in, &vols);

        int size = strikes.size();
        vector<Option> options;
        options.reserve(size);

        for (int i = 0; i < size; ++i)
        {
            Option o;
            o.strike_price = strikes.at(i);
            o.maturity = maturities.at(i);
            o.num_of_terms = num_of_terms.at(i);
            o.reversion_rate = rrps.at(i);
            o.volatility = vols.at(i);
            options.push_back(o);
        }
        return options;
    }
};

#endif