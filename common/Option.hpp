#ifndef OPTION_HPP
#define OPTION_HPP

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

    static vector<Option> read_options()
    {
        vector<real> strikes;
        vector<real> maturities;
        vector<int> num_of_terms;
        vector<real> rrps;
        vector<real> vols;

        FutharkArrays::read_futhark_array(&strikes);
        FutharkArrays::read_futhark_array(&maturities);
        FutharkArrays::read_futhark_array(&num_of_terms);
        FutharkArrays::read_futhark_array(&rrps);
        FutharkArrays::read_futhark_array(&vols);

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