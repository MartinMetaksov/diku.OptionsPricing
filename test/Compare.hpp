#ifndef COMPARE_HPP
#define COMPARE_HPP

#include "catch.hpp"
#include "../common/Real.hpp"
#include <vector>

#ifdef USE_DOUBLE
#define EPSILON std::numeric_limits<double>::epsilon()
#else
#define EPSILON std::numeric_limits<float>::epsilon()
#endif

void compareVectors(std::vector<trinom::real> test, std::vector<trinom::real> gold)
{
    Approx approx = Approx::custom().epsilon(EPSILON).scale(1000);

    REQUIRE(test.size() == gold.size());

    for (auto i = 0; i < test.size(); i++)
    {
        CHECK(test[i] == approx(gold[i]));
    }
}

#endif
