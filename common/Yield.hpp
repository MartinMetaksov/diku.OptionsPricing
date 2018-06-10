#ifndef YIELD_HPP
#define YIELD_HPP

#include <fstream>
#include <stdexcept>
#include <vector>

#include "Arrays.hpp"
#include "Real.hpp"

namespace trinom
{

struct Yield
{
    int N;
    std::vector<real> Prices;
    std::vector<int32_t> TimeSteps;

    Yield(const std::string &filename)
    {
        if (filename.empty())
        {
            throw std::invalid_argument("File not specified.");
        }

        std::ifstream in(filename);

        if (!in)
        {
            throw std::invalid_argument("File does not exist.");
        }

        Arrays::read_array(in, Prices);
        Arrays::read_array(in, TimeSteps);
        N = Prices.size();

        in.close();
    }
};
} // namespace trinom

#endif
