#ifndef COMPARE_HPP
#define COMPARE_HPP

#include "catch.hpp"
#include "../common/Real.hpp"
#include <vector>

#ifdef USE_DOUBLE
#define EPSILON std::numeric_limits<double>::epsilon() * 1000
#else
#define EPSILON std::numeric_limits<float>::epsilon() * 1000
#endif

void compareVectors(std::vector<trinom::real> test, std::vector<trinom::real> gold)
{
    Approx approx = Approx::custom().margin(EPSILON);

    REQUIRE(test.size() == gold.size());

    for (auto i = 0; i < test.size(); i++)
    {
        CHECK(test[i] == approx(gold[i]));
    }
}

#endif
