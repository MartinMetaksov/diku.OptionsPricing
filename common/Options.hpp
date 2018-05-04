#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include <stdexcept>
#include <fstream>
#include <vector>
#include <cinttypes>
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

struct Options
{
    int N;
    vector<float> StrikePrices;
    vector<float> Maturities;
    vector<float> Lengths;
    vector<uint16_t> TermUnits;
    vector<uint16_t> TermStepCounts;
    vector<float> ReversionRates;
    vector<float> Volatilities;
    vector<OptionType> Types;

    Options(const string &filename)
    {
        if (filename.empty())
        {
            throw invalid_argument("File not specified.");
        }

        ifstream in(filename);

        if (!in)
        {
            throw invalid_argument("File '" + filename + "' does not exist.");
        }
        
        Arrays::read_array(in, StrikePrices);
        Arrays::read_array(in, Maturities);
        Arrays::read_array(in, Lengths);
        Arrays::read_array(in, TermUnits);
        Arrays::read_array(in, TermStepCounts);
        Arrays::read_array(in, ReversionRates);
        Arrays::read_array(in, Volatilities);
        Arrays::read_array(in, Types);
        N = StrikePrices.size();

        in.close();
    }

    static void writeOptions(const string &filename, const Options &options)
    {
        if (filename.empty())
        {
            throw invalid_argument("File not specified.");
        }

        ofstream out(filename);

        if (!out)
        {
            throw invalid_argument("File does not exist.");
        }

        Arrays::write_array(out, options.StrikePrices);
        Arrays::write_array(out, options.Maturities);
        Arrays::write_array(out, options.Lengths);
        Arrays::write_array(out, options.TermUnits);
        Arrays::write_array(out, options.TermStepCounts);
        Arrays::write_array(out, options.ReversionRates);
        Arrays::write_array(out, options.Volatilities);
        Arrays::write_array(out, options.Types);

        out.close();
    }
};
}

#endif