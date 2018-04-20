#ifndef OPTION_HPP
#define OPTION_HPP

#include <stdexcept>
#include <fstream>
#include <vector>
#include "Real.hpp"
#include "Arrays.hpp"

using namespace std;

namespace trinom
{

enum class OptionType : char
{
    PUT = 'P',
    CALL = 'C'
};

inline ostream &operator<<(ostream &os, const OptionType t)
{
    os << static_cast<char>(t);
    return os;
}

inline istream &operator>>(istream &is, OptionType &t)
{
    char c;
    is >> c;
    t = static_cast<OptionType>(c);
    if (OptionType::CALL != t && OptionType::PUT != t)
    {
        throw out_of_range("Invalid OptionType read from stream.");
    }
    return is;
}

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
    OptionType Type;

    static vector<Option> readOptions(const string &filename)
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
        vector<OptionType> types;

        Arrays::read_array(in, &strikes);
        Arrays::read_array(in, &maturities);
        Arrays::read_array(in, &lengths);
        Arrays::read_array(in, &termunits);
        Arrays::read_array(in, &termsteps);
        Arrays::read_array(in, &rrps);
        Arrays::read_array(in, &vols);
        Arrays::read_array(in, &types);

        in.close();

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
            o.Type = types.at(i);
            options.push_back(o);
        }
        return options;
    }
};
}

#endif