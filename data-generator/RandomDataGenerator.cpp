// basic file operations
#include <iostream>
#include <fstream>

#include "../common/Args.hpp"
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
    auto args = Args::parseArgs(argc, argv);
    auto numOptions = args.numOptions;
    // vector<Option> options;

    ofstream myfile;
    string filename = "random-" + to_string(numOptions) + ".in";
    string path = "../data/";

    cout << "filename: " << path + filename << endl;
    myfile.open(path + filename);
    for (int i = 0; i < numOptions; i++)
    {
        // Option o;
        // o.strikePrice = 63;

        int strikePrice = 63;
        int maturity = 9 * randIntInRange(1, 2); // random int between 1-2; controls the height
        int length = 3;
        int termUnit = 365;
        int termStepCount = randIntInRange(5, 50);      // random int between 5-50; controls both the width & height                           // controls both the width & height, but I left this unchanged for consistency
        real reversionRate = randRealInRange(0.1, 0.9); // random real between 0.1-0.9; controls the width
        real volatility = 0.01;
        char type = 'P';

        // options.push(o);
    }

    myfile.close();
    return 0;
}