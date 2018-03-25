#include "catch.hpp"
#include "Mock.hpp"
#include "../seq/Seq.hpp"
#include "../cuda-option/CudaOption.cuh"

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
        bookResults.push_back(Seq::computeSingleOption(book[i]));
    }

    SECTION("Compute one book option")
    {
        real result;
        computeCuda(book, &result, 1);

        REQUIRE(result == bookResults.at(0));
    }
    SECTION("Compute book options with more precision")
    {
        vector<real> results;
        results.resize(bookCount);
        computeCuda(book, results.data(), bookCount);

        REQUIRE(bookResults == results);
    }
    SECTION("Compute random options")
    {
        // Make mock constants, count should be a multiple of 4
        int count = 104;
        OptionConstants options[count];
        vector<real> goldResults;
        goldResults.reserve(count);
        for (auto i = 0; i < count; i += 4)
        {
            Mock::mockConstants(options + i + 1, 1, 101, 12000);
            Mock::mockConstants(options + i, 1, 10001, 1200);
            Mock::mockConstants(options + i + 2, 1, 11, 1800);
            Mock::mockConstants(options + i + 3, 1, 1001, 12);
        }
        for (auto i = 0; i < count; ++i)
        {
            bookResults.push_back(Seq::computeSingleOption(options[i]));
        }

        vector<real> results;
        results.resize(count);
        computeCuda(book, results.data(), count);

        REQUIRE(goldResults == results);
    }
}
