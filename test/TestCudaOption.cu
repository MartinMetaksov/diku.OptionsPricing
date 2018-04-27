#include "catch.hpp"
#include "../cuda-option/CudaOption.cuh"
#include "../seq/Seq.hpp"
#include "Mock.hpp"

using namespace trinom;

#define YIELD_CURVE_PATH "../data/yield.in" 

void compareVectors(vector<real> test, vector<real> gold)
{
    REQUIRE(test.size() == gold.size());

    for (auto i = 0; i < test.size(); i++)
    {
        // epsilon serves to set the percentage by which a result can be erroneous, before it is rejected.
        CHECK(test[i] == Approx(gold[i]).epsilon(0.0001));
    }
}

TEST_CASE("One option per thread cuda")
{
    auto yield = Yield::readYieldCurve(YIELD_CURVE_PATH);

    int bookCount = 100;
    vector<real> bookResults;
    vector<OptionConstants> book;
    bookResults.reserve(bookCount);
    book.reserve(bookCount);
    for (int i = 0; i < bookCount; ++i)
    {
        Option o;
        o.Length = 3;
        o.Maturity = 9;
        o.StrikePrice = 63;
        o.TermUnit = 365;
        o.TermStepCount = i + 1;
        o.ReversionRate = 0.1;
        o.Volatility = 0.01;

        book.push_back(OptionConstants::computeConstants(o));
        bookResults.push_back(seq::computeSingleOption(book[i], yield));
    }

    SECTION("Compute book options with more precision")
    {
        vector<real> results;
        results.resize(bookCount);
        cuda::computeOptionsCoalesced(book, yield, results.data());

        compareVectors(results, bookResults);
    }
    SECTION("Compute random options")
    {
        int count = 104;
        vector<OptionConstants> options;
        options.resize(count);

        Mock::mockConstants(options.data(), count, 1001, 12);

        vector<real> goldResults;
        goldResults.reserve(count);
        for (auto &option : options)
        {
            goldResults.push_back(seq::computeSingleOption(option, yield));
        }

        vector<real> results;
        results.resize(count);
        cuda::computeOptionsCoalesced(options, yield, results.data());

        compareVectors(results, goldResults);
    }
}
