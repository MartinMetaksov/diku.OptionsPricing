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

enum class OptionType : int8_t
{
    PUT = 0,
    CALL = 1
};

inline ostream &operator<<(ostream &os, const OptionType t)
{
    os << static_cast<int>(t);
    return os;
}

inline istream &operator>>(istream &is, OptionType &t)
{
    int c;
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
    vector<real> StrikePrices;
    vector<real> Maturities;
    vector<real> Lengths;
    vector<uint16_t> TermUnits;
    vector<uint16_t> TermStepCounts;
    vector<real> ReversionRates;
    vector<real> Volatilities;
    vector<OptionType> Types;

    Options(const int count)
    {
        N = count;
        Lengths.reserve(N);
        Maturities.reserve(N);
        StrikePrices.reserve(N);
        TermUnits.reserve(N);
        TermStepCounts.reserve(N);
        ReversionRates.reserve(N);
        Volatilities.reserve(N);
        Types.reserve(N);
    }

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

    void writeToFile(const string &filename)
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

        Arrays::write_array(out, StrikePrices);
        Arrays::write_array(out, Maturities);
        Arrays::write_array(out, Lengths);
        Arrays::write_array(out, TermUnits);
        Arrays::write_array(out, TermStepCounts);
        Arrays::write_array(out, ReversionRates);
        Arrays::write_array(out, Volatilities);
        Arrays::write_array(out, Types);

        out.close();
    }
};
} // namespace trinom

#endif