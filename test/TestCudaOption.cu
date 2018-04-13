#include "catch.hpp"
#include "../cuda-option/CudaOption.cuh"
#include "../seq/Seq.hpp"
#include "Mock.hpp"

using namespace trinom;

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
    int bookCount = 100;
    OptionConstants book[bookCount];
    vector<real> bookResults;
    bookResults.reserve(bookCount);
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

        book[i] = OptionConstants::computeConstants(o);
        bookResults.push_back(seq::computeSingleOption(book[i]));
    }

    SECTION("Compute one book option")
    {
        real result;
        cuda::computeOptions(book, &result, 1);

        REQUIRE(result == Approx(bookResults.at(0)));
    }
    SECTION("Compute book options with more precision")
    {
        vector<real> results;
        results.resize(bookCount);
        cuda::computeOptions(book, results.data(), bookCount);

        compareVectors(results, bookResults);
    }
    SECTION("Compute random options")
    {
        int count = 104;
        OptionConstants options[count];
        OptionConstants *options_p = options;

        Mock::mockConstants(options_p, count, 1001, 12);

        vector<real> goldResults;
        goldResults.reserve(count);
        for (auto &option : options)
        {
            goldResults.push_back(seq::computeSingleOption(option));
        }

        vector<real> results;
        results.resize(count);
        cuda::computeOptions(options_p, results.data(), count);

        compareVectors(results, goldResults);
    }
}
