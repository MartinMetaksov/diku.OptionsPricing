// basic file operations
#include <iostream>
#include <fstream>

#include "../common/getoptpp/getopt_pp_standalone.h"
#include "../common/Option.hpp"
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

    vector<Option> options;
    options.reserve(numOptions);

    for (int i = 0; i < numOptions; i++)
    {
        Option o;
        o.StrikePrice = 63;
        o.Maturity = 9 * randIntInRange(1, 2); // random int between 1-2; controls the height
        o.Length = 3;
        o.TermUnit = 365;
        o.TermStepCount = randIntInRange(5, 50);      // random int between 5-50; controls both the width & height
        // controls both the width & height, but I left this unchanged for consistency
        o.ReversionRate = randRealInRange(0.1, 0.9); // random real between 0.1-0.9; controls the width
        o.Volatility = 0.01;
        o.Type = OptionType::PUT;
        options.push_back(o);
    }

    cout << "filename: " << filename << endl;

    Option::writeOptions(filename, options);

    return 0;
}