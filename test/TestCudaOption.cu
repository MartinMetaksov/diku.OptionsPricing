#include "catch.hpp"

#define USE_DOUBLE
#include "../cuda-option/Version1.cuh"
#include "../cuda-option/Version2.cuh"
#include "../cuda-option/Version3.cuh"
#include "../cuda-multi/Version1.cuh"
#include "../cuda-multi/Version2.cuh"
#include "../cuda-multi/Version3.cuh"
#include "../seq/Seq.hpp"

using namespace std;
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

TEST_CASE("Book options")
{
    Yield yield(YIELD_CURVE_PATH);
    
    Options options(100);
    for (int i = 1; i <= options.N; ++i)
    {
        options.Lengths.push_back(3);
        options.Maturities.push_back(9);
        options.StrikePrices.push_back(63);
        options.TermUnits.push_back(365);
        options.TermStepCounts.push_back(i);
        options.ReversionRates.push_back(0.1);
        options.Volatilities.push_back(0.01);
        options.Types.push_back(OptionType::PUT);
    }
    
    vector<real> seqResults, cudaResults;
    seqResults.reserve(options.N);
    seq::computeOptions(options, yield, seqResults);

    SECTION("CUDA option version 1")
    {
        vector<real> results;
        results.resize(options.N);
        cuda::option::KernelRunNaive kernelRun;
        kernelRun.run(options, yield, results, 64);
        compareVectors(results, seqResults);
    }

    SECTION("CUDA option version 2")
    {
        vector<real> results;
        results.resize(options.N);
        cuda::option::KernelRunCoalesced kernelRun;
        kernelRun.run(options, yield, results, 64);
        compareVectors(results, seqResults);
    }

    SECTION("CUDA option version 3")
    {
        vector<real> results;
        results.resize(options.N);
        cuda::option::KernelRunCoalescedChunk kernelRun(64);
        kernelRun.run(options, yield, results, 64);
        compareVectors(results, seqResults);
    }

    SECTION("CUDA option version 4")
    {
        vector<real> results;
        results.resize(options.N);
        cuda::option::KernelRunCoalescedChunk kernelRun(32);
        kernelRun.run(options, yield, results, 64);
        compareVectors(results, seqResults);
    }

    SECTION("CUDA multi version 1")
    {
        vector<real> results;
        results.resize(options.N);
        cuda::multi::KernelRunNaive kernelRun;
        kernelRun.run(options, yield, results, 512);
        compareVectors(results, seqResults);
    }

    SECTION("CUDA multi version 2")
    {
        vector<real> results;
        results.resize(options.N);
        cuda::multi::KernelRunCoalesced kernelRun;
        kernelRun.run(options, yield, results, 512);
        compareVectors(results, seqResults);
    }

    SECTION("CUDA multi version 3")
    {
        vector<real> results;
        results.resize(options.N);
        cuda::multi::KernelRunCoalescedBlock kernelRun;
        kernelRun.run(options, yield, results, 512);
        compareVectors(results, seqResults);
    }
}
