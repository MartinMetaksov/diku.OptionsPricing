// basic file operations
#include <iostream>
#include <fstream>

#include "../common/getoptpp/getopt_pp_standalone.h"
#include "../common/Options.hpp"
#include "../common/Real.hpp"

using namespace std;
using namespace trinom;

real randRealInRange(real a, real b)
{
    real random = ((real)rand()) / (real)RAND_MAX;
    real diff = b - a;
    real r = random * diff;
    return a + r;
}

int randIntInRange(int a, int b)
{
    return rand() % (b - a + 1) + a;
}

int main(int argc, char *argv[])
{
    int numOptions;
    GetOpt::GetOpt_pp cmd(argc, argv);
    cmd >> GetOpt::Option('n', "numOptions", numOptions);

    ofstream myfile;
    string filename = "../data/random-" + to_string(numOptions) + ".in";

    Options options(numOptions);

    for (int i = 0; i < numOptions; i++)
    {
        options.Lengths.push_back(3);
        options.Maturities.push_back(9 * randIntInRange(1, 2)); // random int between 1-2; controls the height
        options.StrikePrices.push_back(63);
        options.TermUnits.push_back(365);
        options.TermStepCounts.push_back(randIntInRange(5, 50)); // random int between 5-50; controls both the width & height
        // controls both the width & height, but I left this unchanged for consistency
        options.ReversionRates.push_back(randRealInRange(0.1, 0.9)); // random real between 0.1-0.9; controls the width
        options.Volatilities.push_back(0.01);
        options.Types.push_back(OptionType::PUT);
    }

    cout << "filename: " << filename << endl;

    options.writeToFile(filename);

    return 0;
}