// basic file operations
#include <iostream>
#include <fstream>

// math
#include <math.h>
#include <algorithm>
#include <random>

#include "../common/getoptpp/getopt_pp_standalone.h"
#include "../common/Options.hpp"
#include "../common/Real.hpp"

using namespace std;
using namespace trinom;

struct RandOption
{
    real Maturity;
    int TermStepCount;
    real ReversionRate;
    int Width;
    int Height;

    RandOption(const real maturity, const int termStepCount, const real reversionRate)
    {
        Maturity = maturity;
        TermStepCount = termStepCount;
        ReversionRate = reversionRate;
        computeWidth();
        computeHeight();
    }

    void computeWidth()
    {
        Width = 2 * ((int)(minus184 / (exp(-ReversionRate * ((ceil((real)year / 365)) / (real)TermStepCount)) - one)) + 1) + 1;
    }

    void computeHeight()
    {
        Height = (TermStepCount * (ceil((real)year / 365)) * Maturity) + 1;
    }
};

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

void addOption(vector<RandOption> &options, const RandOption o, int &currentTotalHeight, int &currentTotalWidth)
{
    currentTotalHeight += o.Height;
    currentTotalWidth += o.Width;
    options.push_back(o);
}

void genOneLargeOption(vector<RandOption> &options, int &totalHeight, int &totalWidth, int &currentTotalHeight, int &currentTotalWidth)
{
    RandOption o(4 * 9, 277, 0.1);
    addOption(options, o, currentTotalHeight, currentTotalWidth);
    totalWidth -= o.Width;
    totalHeight -= o.Height;

    while (totalHeight / (real)totalWidth < 2.437)
    {
        RandOption of(0, 277, 0.1);
        addOption(options, of, currentTotalHeight, currentTotalWidth);
        totalWidth -= of.Width;
        totalHeight -= of.Height;
    }
}

void genMultLargeOptions(vector<RandOption> &options, int &totalHeight, int &totalWidth, int &currentTotalHeight, int &currentTotalWidth)
{
    int loHeight = 0;

    while (loHeight < totalHeight / 2)
    {
        RandOption o(4 * 9, 277, 0.1);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
        loHeight += o.Height;
        totalWidth -= o.Width;
        totalHeight -= o.Height;
    }

    while (totalHeight / (float)totalWidth < 2.437)
    {
        RandOption o(0, 277, 0.1);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
        totalWidth -= o.Width;
        totalHeight -= o.Height;
    }
}

int sumWidth(const int x, const RandOption y) { return x + y.Width; }
int sumHeight(const int x, const RandOption y) { return x + y.Height; }

int getFinalWidth(vector<RandOption> &options)
{
    return accumulate(options.begin(), options.end(), 0, sumWidth);
}
int getFinalHeight(vector<RandOption> &options)
{
    return accumulate(options.begin(), options.end(), 0, sumHeight);
}

int main(int argc, char *argv[])
{
    int type;
    int totalHeight;
    string filename;
    GetOpt::GetOpt_pp cmd(argc, argv);
    // 0 - uniform; 1 - one large; 2 - mix of large and small; 3 - scrambled, large differences b/w widths and heights
    cmd >> GetOpt::Option('t', "type", type);
    cmd >> GetOpt::Option('f', "fileName", filename);
    cmd >> GetOpt::Option('h', "totalHeight", totalHeight);

    // ofstream myfile;
    filename = "../data/" + filename + ".in";
    int totalWidth = (int)ceil(totalHeight / 2.437);
    cout << "total allowed width: " << totalWidth << endl;
    cout << "total allowed height: " << totalHeight << endl;

    int stepMin = 1;
    int stepMax = 277; // max allowed term ste count in order to keep tree widths < 1024
    int currentTotalHeight = 0;
    int currentTotalWidth = 0;
    int maxWidth = 1021;

    vector<RandOption> randOptions;

    // type == 0 is the default, no actions needed
    if (type == 1)
    {
        genOneLargeOption(randOptions, totalHeight, totalWidth, currentTotalHeight, currentTotalWidth);
    }
    else if (type == 2)
    {
        genMultLargeOptions(randOptions, totalHeight, totalWidth, currentTotalHeight, currentTotalWidth);
    }

    while (true)
    {
        try
        {
            vector<RandOption> tempOptions;
            while (currentTotalHeight < totalHeight)
            {
                // generate option
                RandOption o(type == 3 ? randIntInRange(1, 18) : 9,
                             randIntInRange(stepMin, stepMax),
                             type == 3 ? randRealInRange(0.05, 0.17) : 0.1);

                if (currentTotalHeight + o.Height > totalHeight)
                {
                    // width filler option
                    RandOption widthFiller(0, stepMax, 0.1);
                    while (totalWidth - currentTotalWidth > maxWidth)
                    {
                        if (currentTotalWidth + widthFiller.Width > totalWidth)
                        {
                            break;
                        }
                        addOption(tempOptions, widthFiller, currentTotalHeight, currentTotalWidth);
                    }
                    break;
                }
                addOption(tempOptions, o, currentTotalHeight, currentTotalWidth);
            }

            int remainingWidth = totalWidth - currentTotalWidth;
            int remainingHeight = totalHeight - currentTotalHeight;
            if (remainingWidth < 0 || remainingHeight < 0)
            {
                throw 100; // invalid total width or total height
            }
            else if (remainingWidth % 2 == 0)
            {
                throw 101; // remaining width cannot be filled
            }

            real rr = 0.1;
            for (real i = 0.1; i > 0; i -= 0.00001)
            {
                // width filler option
                int maturity = 1;
                RandOption o(1, 1, i);

                if (o.Width == remainingWidth)
                {
                    rr = i;
                    break;
                }
                if (o.Width > maxWidth)
                {
                    throw 102; //could not find a suitable width for the fill
                }
            }
            RandOption o(remainingHeight - 1, 1, rr);
            addOption(tempOptions, o, currentTotalHeight, currentTotalWidth);
            randOptions.insert(randOptions.end(), tempOptions.begin(), tempOptions.end());
            break;
        }
        catch (int e)
        {
            currentTotalHeight = 0;
            currentTotalWidth = 0;
        }
    }

    cout << "final total width: " << getFinalWidth(randOptions) << endl;
    cout << "final total height: " << getFinalHeight(randOptions) << endl;
    cout << "total options: " << randOptions.size() << endl;

    auto rng = default_random_engine{};
    shuffle(begin(randOptions), end(randOptions), rng);

    Options options(randOptions.size());
    for (int i = 0; i < randOptions.size(); i++)
    {
        options.Lengths.push_back(3);
        options.Maturities.push_back(randOptions.at(i).Maturity);
        options.StrikePrices.push_back(63);
        options.TermUnits.push_back(365);
        options.TermStepCounts.push_back(randOptions.at(i).TermStepCount);
        options.ReversionRates.push_back(randOptions.at(i).ReversionRate);
        options.Volatilities.push_back(0.01);
        options.Types.push_back(OptionType::PUT);
    }

    cout << "writing to file: " << filename << endl;
    options.writeToFile(filename);
    return 0;
}