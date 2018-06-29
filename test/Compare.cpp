#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "Compare.hpp"
#include "../common/Arrays.hpp"

using namespace trinom;

TEST_CASE("Equality of two arrays")
{
    std::vector<real> array1;
    Arrays::read_array(std::cin, array1);

    std::vector<real> array2;
    Arrays::read_array(std::cin, array2);

    compareVectors(array1, array2);
}
