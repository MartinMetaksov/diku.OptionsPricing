#include "catch.hpp"

#define USE_DOUBLE
#include "../cuda-option/Version1.cuh"
#include "../cuda-option/Version2.cuh"
#include "../cuda-option/Version3.cuh"
#include "../seq/Seq.hpp"

using namespace trinom;

#define YIELD_CURVE_PATH "../data/yield.in" 

void compareVectors(vector<real> test, vector<real> gold)
{
    REQUIRE(test.size() == gold.size());

    for (auto i = 0; i < test.size(); i++)
    {
        CHECK(test[i] == Approx(gold[i]));
    }
}

TEST_CASE("One option per thread cuda")
{
    Yield yield(YIELD_CURVE_PATH);

    Options options(100);
    for (int i = 0; i < options.N; ++i)
    {
        options.Lengths.push_back(3);
        options.Maturities.push_back(9);
        options.StrikePrices.push_back(63 + i * 20);
        options.TermUnits.push_back(365);
        options.TermStepCounts.push_back(i + 1);
        options.ReversionRates.push_back(0.1);
        options.Volatilities.push_back(0.01);
        options.Types.push_back(i % 2 == 0 ? OptionType::PUT : OptionType::CALL);
    }
    
    vector<real> seqResults, cudaResults;
    seqResults.reserve(options.N);
    seq::computeOptions(options, yield, seqResults);

    SECTION("Version 1")
    {
        vector<real> results;
        results.resize(options.N);
        cuda::computeOptionsNaive(options, yield, results);
        compareVectors(results, seqResults);
    }

    SECTION("Version 2")
    {
        vector<real> results;
        results.resize(options.N);
        cuda::computeOptionsCoalesced(options, yield, results);
        compareVectors(results, seqResults);
    }

    SECTION("Version 3")
    {
        vector<real> results;
        results.resize(options.N);
        cuda::computeOptionsWithPaddingPerThreadBlock(options, yield, results);
        compareVectors(results, seqResults);
    }
}
