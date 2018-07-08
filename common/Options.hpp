#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include <cinttypes>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "Arrays.hpp"
#include "Real.hpp"

namespace trinom
{

enum class SortType : char
{
    WIDTH_DESC = 'W',
    WIDTH_ASC = 'w',
    HEIGHT_DESC = 'H',
    HEIGHT_ASC = 'h',
    NONE = '-'
};

enum class OptionType : int8_t
{
    CALL = 0,
    PUT = 1
};

inline std::ostream &operator<<(std::ostream &os, const OptionType t)
{
    os << static_cast<int>(t);
    return os;
}

inline std::istream &operator>>(std::istream &is, OptionType &t)
{
    int c;
    is >> c;
    t = static_cast<OptionType>(c);
    if (OptionType::CALL != t && OptionType::PUT != t)
    {
        throw std::out_of_range("Invalid OptionType read from stream.");
    }
    return is;
}

struct Options
{
    int N;
    std::vector<real> StrikePrices;
    std::vector<real> Maturities;
    std::vector<real> Lengths;
    std::vector<uint16_t> TermUnits;
    std::vector<uint16_t> TermStepCounts;
    std::vector<real> ReversionRates;
    std::vector<real> Volatilities;
    std::vector<OptionType> Types;

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

    Options(const std::string &filename)
    {
        if (filename.empty())
        {
            throw std::invalid_argument("File not specified.");
        }

        std::ifstream in(filename);

        if (!in)
        {
            throw std::invalid_argument("File '" + filename + "' does not exist.");
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

    void writeToFile(const std::string &filename)
    {
        if (filename.empty())
        {
            throw std::invalid_argument("File not specified.");
        }

        std::ofstream out(filename);

        if (!out)
        {
            throw std::invalid_argument("File does not exist.");
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