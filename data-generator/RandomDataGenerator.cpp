// basic file operations
#include <iostream>
#include <fstream>

// math
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

#include "../common/getoptpp/getopt_pp_standalone.h"
#include "../common/Options.hpp"
#include "../common/Real.hpp"

using namespace std;
using namespace trinom;

struct RandOption
{
    int Maturity;
    int TermStepCount;
    real ReversionRate;
    int Width;
    int Height;
    bool Skewed;

    RandOption(const int maturity, const int termStepCount, const real reversionRate, const bool skewed)
    {
        Maturity = maturity;
        TermStepCount = termStepCount;
        ReversionRate = reversionRate;
        Skewed = skewed;
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

void addOption(vector<RandOption> &options, const RandOption o, long &currentTotalHeight, long &currentTotalWidth)
{
    currentTotalHeight += o.Height;
    currentTotalWidth += o.Width;
    options.push_back(o);
}

long sumSkewedWidth(const long x, const RandOption y) { return y.Skewed ? x + y.Width : x; }
long sumSkewedHeight(const long x, const RandOption y) { return y.Skewed ? x + y.Height : x; }
int getNumSkewed(const int x, const RandOption y) { return y.Skewed ? x + 1 : x; }
long sumWidth(const long x, const RandOption y) { return x + y.Width; }
long sumHeight(const long x, const RandOption y) { return x + y.Height; }

long getFinalWidth(vector<RandOption> &options)
{
    return accumulate(options.begin(), options.end(), 0, sumWidth);
}
long getFinalHeight(vector<RandOption> &options)
{
    return accumulate(options.begin(), options.end(), 0, sumHeight);
}
long getFinalSkewedWidth(vector<RandOption> &options)
{
    return accumulate(options.begin(), options.end(), 0, sumSkewedWidth);
}
long getFinalSkewedHeight(vector<RandOption> &options)
{
    return accumulate(options.begin(), options.end(), 0, sumSkewedHeight);
}

int getNumSkewedOptions(vector<RandOption> &options)
{
    return accumulate(options.begin(), options.end(), 0, getNumSkewed);
}

void writeOptionsToFile(vector<RandOption> &randOptions,
                        const string filename,
                        const int dataType,
                        const long constProduct,
                        const int skewPercent)
{
    long finalWidth = getFinalWidth(randOptions);
    long finalHeight = getFinalHeight(randOptions);

    string dataFile = "../data/" + filename + ".in";
    string markdownFile = "../data/dataInfo/" + filename + ".md";

    std::ofstream mdFile(markdownFile);

    mdFile << "Filename: " << dataFile << endl;
    mdFile << "Total size: " << randOptions.size() << endl;
    mdFile << "Current total width: " << finalWidth << endl;
    mdFile << "Current total height: " << finalHeight << endl;
    mdFile << "Constant max product: " << constProduct << endl;
    mdFile << "Current product: " << finalWidth * finalHeight << endl;
    mdFile << "Deviation: " << abs(constProduct - (finalWidth * finalHeight)) * 100 / (real)constProduct << "%" << endl;
    if (dataType == 4 || dataType == 5 || dataType == 6)
    {
        mdFile << "Skew: " << skewPercent << "%" << endl;
        mdFile << "Total skewed options: " << getNumSkewedOptions(randOptions) << endl;
        mdFile << "Skewed total width: " << getFinalSkewedWidth(randOptions) << endl;
        mdFile << "Skewed total height: " << getFinalSkewedHeight(randOptions) << endl;
    }

    mdFile.close();

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

    options.writeToFile(dataFile);
}

void distribute_0(vector<RandOption> &options, const long constProduct)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;
    while ((currentTotalHeight * currentTotalWidth) < constProduct)
    {
        RandOption o(9, 12, 0.1, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }
    writeOptionsToFile(options, "0_UNIFORM", 0, constProduct, 0);
}

void distribute_1(vector<RandOption> &options, const long constProduct)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;

    while ((currentTotalHeight * currentTotalWidth) < constProduct)
    {
        int maturity = randIntInRange(1, 729);
        real reversionRate = randRealInRange(0.00433, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    writeOptionsToFile(options, "1_RAND", 1, constProduct, 0);
}

void distribute_2(vector<RandOption> &options, const long constProduct)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;

    while ((currentTotalHeight * currentTotalWidth) < constProduct)
    {
        real reversionRate = randRealInRange(0.00433, 0.99);
        RandOption o(9, 12, reversionRate, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    writeOptionsToFile(options, "2_RANDCONSTHEIGHT", 2, constProduct, 0);
}

void distribute_3(vector<RandOption> &options, const long constProduct)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;

    while ((currentTotalHeight * currentTotalWidth) < constProduct)
    {
        int maturity = randIntInRange(1, 729);
        RandOption o(maturity, 12, 0.1, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    writeOptionsToFile(options, "3_RANDCONSTWIDTH", 3, constProduct, 0);
}

void distribute_4(vector<RandOption> &options, const long constProduct, const int skewPerc)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;

    while ((currentTotalHeight * currentTotalWidth) < (skewPerc / (real)100) * constProduct)
    {
        int maturity = randIntInRange(648, 729);
        real reversionRate = randRealInRange(0.00433, 0.0045);
        RandOption o(maturity, 12, reversionRate, true);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    while ((currentTotalHeight * currentTotalWidth) < constProduct)
    {
        int maturity = randIntInRange(1, 81);
        real reversionRate = randRealInRange(0.01, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    writeOptionsToFile(options, "4_SKEWED", 4, constProduct, skewPerc);
}

void distribute_5(vector<RandOption> &options, const long constProduct, const int skewPerc)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;

    while ((currentTotalHeight * currentTotalWidth) < (skewPerc / (real)100) * constProduct)
    {
        real reversionRate = randRealInRange(0.00433, 0.0045);
        RandOption o(9, 12, reversionRate, true);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    while ((currentTotalHeight * currentTotalWidth) < constProduct)
    {
        int maturity = randIntInRange(1, 81);
        real reversionRate = randRealInRange(0.01, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    writeOptionsToFile(options, "5_SKEWEDCONSTHEIGHT", 5, constProduct, skewPerc);
}

void distribute_6(vector<RandOption> &options, const long constProduct, const int skewPerc)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;

    while ((currentTotalHeight * currentTotalWidth) < (skewPerc / (real)100) * constProduct)
    {
        int maturity = randIntInRange(648, 729);
        RandOption o(maturity, 12, 0.1, true);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    while ((currentTotalHeight * currentTotalWidth) < constProduct)
    {
        int maturity = randIntInRange(1, 81);
        real reversionRate = randRealInRange(0.01, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    writeOptionsToFile(options, "6_SKEWEDCONSTWIDTH", 6, constProduct, skewPerc);
}

int main(int argc, char *argv[])
{
    int dataType;
    long totalHeight;
    long totalWidth;
    int skewPercent = 1;
    GetOpt::GetOpt_pp cmd(argc, argv);

    // 0 - uniform;
    // 1 - random;
    // 2 - random fixed height;
    // 3 - random fixed width;
    // 4 - skewed;
    // 5 - skewed fixed height;
    // 6 - skewed fixed width;
    cmd >> GetOpt::Option('t', "type", dataType);
    cmd >> GetOpt::Option('h', "totalHeight", totalHeight);
    cmd >> GetOpt::Option('w', "totalWidth", totalWidth);
    cmd >> GetOpt::Option('s', "skewPerc", skewPercent);

    // ofstream myfile;
    long constProduct = totalWidth * totalHeight;
    vector<RandOption> randOptions;

    switch (dataType)
    {
    case 0:
        distribute_0(randOptions, constProduct);
        break;
    case 1:
        distribute_1(randOptions, constProduct);
        break;
    case 2:
        distribute_2(randOptions, constProduct);
        break;
    case 3:
        distribute_3(randOptions, constProduct);
        break;
    case 4:
        distribute_4(randOptions, constProduct, skewPercent);
        break;
    case 5:
        distribute_5(randOptions, constProduct, skewPercent);
        break;
    case 6:
        distribute_6(randOptions, constProduct, skewPercent);
        break;
    default:
        // if out of range - just print them all
        vector<RandOption> rand0;
        vector<RandOption> rand1;
        vector<RandOption> rand2;
        vector<RandOption> rand3;
        vector<RandOption> rand4;
        vector<RandOption> rand5;
        vector<RandOption> rand6;
        distribute_0(rand0, constProduct);
        distribute_1(rand1, constProduct);
        distribute_2(rand2, constProduct);
        distribute_3(rand3, constProduct);
        distribute_4(rand4, constProduct, skewPercent);
        distribute_5(rand5, constProduct, skewPercent);
        distribute_6(rand6, constProduct, skewPercent);
    }

    return 0;
}