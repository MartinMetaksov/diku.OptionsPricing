#ifndef MOCK_HPP
#define MOCK_HPP

#include "../common/OptionConstants.hpp"

using namespace trinom;

class Mock
{
  public:
    static void mockConstants(OptionConstants *result, int count, int width, int steps)
    {
        OptionConstants c;
        c.n = steps;
        c.dt = steps / 36.0;
        c.t = steps / 12;
        c.X = 60;
        c.dr = 0.1;
        c.jmax = width / 2;
        c.M = minus184 / (c.jmax - 1);
        c.width = 2 * c.jmax + 1;
        fill_n(result, count, c);
    }
};

#endif
