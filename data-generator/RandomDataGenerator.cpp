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

long getApproxNumOfOperations(long width, long height)
{
    return width * height;
}

real randRealInRange(real a, real b)
{
    random_device rand_dev;
    mt19937 gen(rand_dev());
    uniform_real_distribution<> dis(a, b);
    return dis(gen);
}

int randIntInRange(int a, int b)
{
    return rand() % (b - a + 1) + a;
}

int getWidthByRRandTsc(int termStepCount, real rr)
{
    return 2 * ((int)(minus184 / (exp(-rr * ((ceil((real)year / 365)) / (real)termStepCount)) - one)) + 1) + 1;
}

real getRandRRByWidthRangeTscAndRRRange(int termStepCount, real a, real b)
{
    real rr = zero;
    int targetWidth = -1;
    int widthMin = getWidthByRRandTsc(termStepCount, b);
    int widthMax = getWidthByRRandTsc(termStepCount, a);
    int randWidth = randIntInRange(widthMin, widthMax);

    // set an offset, as can't always get the exact width
    while (targetWidth == -1 ||
           (targetWidth != randWidth && (targetWidth < randWidth - 20 || targetWidth > randWidth + 20)))
    {
        rr = randRealInRange(a, b);
        targetWidth = getWidthByRRandTsc(termStepCount, rr);
    }
    return rr;
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
    long finalNumOperations = getApproxNumOfOperations(finalWidth, finalHeight);
    mdFile << "Current product: " << finalNumOperations << endl;
    mdFile << "Deviation: " << abs(constProduct - finalNumOperations) * 100 / (real)constProduct << "%" << endl;
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

    while (getApproxNumOfOperations(currentTotalWidth, currentTotalHeight) < constProduct)
    {
        RandOption o(9, 12, 0.1, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    string filename = "0_UNIFORM";
    writeOptionsToFile(options, filename, 0, constProduct, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_1(vector<RandOption> &options, const long constProduct)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;

    while (getApproxNumOfOperations(currentTotalWidth, currentTotalHeight) < constProduct)
    {
        int maturity = randIntInRange(1, 729);
        real reversionRate = getRandRRByWidthRangeTscAndRRRange(12, 0.00433, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    string filename = "1_RAND";
    writeOptionsToFile(options, filename, 1, constProduct, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_2(vector<RandOption> &options, const long constProduct)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;

    while (getApproxNumOfOperations(currentTotalWidth, currentTotalHeight) < constProduct)
    {
        real reversionRate = getRandRRByWidthRangeTscAndRRRange(12, 0.00433, 0.99);
        RandOption o(9, 12, reversionRate, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    string filename = "2_RANDCONSTHEIGHT";
    writeOptionsToFile(options, filename, 2, constProduct, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_3(vector<RandOption> &options, const long constProduct)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;

    while (getApproxNumOfOperations(currentTotalWidth, currentTotalHeight) < constProduct)
    {
        int maturity = randIntInRange(1, 729);
        RandOption o(maturity, 12, 0.1, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }
    string filename = "3_RANDCONSTWIDTH";
    writeOptionsToFile(options, filename, 3, constProduct, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_4(vector<RandOption> &options, const long constProduct, const int skewPerc)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;

    while (getApproxNumOfOperations(currentTotalWidth, currentTotalHeight) < (skewPerc / (real)100) * constProduct)
    {
        int maturity = randIntInRange(648, 729);
        real reversionRate = getRandRRByWidthRangeTscAndRRRange(12, 0.00433, 0.0045);
        RandOption o(maturity, 12, reversionRate, true);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    while (getApproxNumOfOperations(currentTotalWidth, currentTotalHeight) < constProduct)
    {
        int maturity = randIntInRange(1, 81);
        real reversionRate = getRandRRByWidthRangeTscAndRRRange(12, 0.01, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    string filename = "4_SKEWED";
    writeOptionsToFile(options, filename, 4, constProduct, skewPerc);
    cout << "finished writing to " << filename << endl;
}

void distribute_5(vector<RandOption> &options, const long constProduct, const int skewPerc)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;

    while (getApproxNumOfOperations(currentTotalWidth, currentTotalHeight) < (skewPerc / (real)100) * constProduct)
    {
        real reversionRate = getRandRRByWidthRangeTscAndRRRange(12, 0.00433, 0.0045);
        RandOption o(9, 12, reversionRate, true);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    while (getApproxNumOfOperations(currentTotalWidth, currentTotalHeight) < constProduct)
    {
        int maturity = randIntInRange(1, 81);
        real reversionRate = getRandRRByWidthRangeTscAndRRRange(12, 0.01, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    string filename = "5_SKEWEDCONSTHEIGHT";
    writeOptionsToFile(options, filename, 5, constProduct, skewPerc);
    cout << "finished writing to " << filename << endl;
}

void distribute_6(vector<RandOption> &options, const long constProduct, const int skewPerc)
{
    long currentTotalWidth = 0;
    long currentTotalHeight = 0;

    while (getApproxNumOfOperations(currentTotalWidth, currentTotalHeight) < (skewPerc / (real)100) * constProduct)
    {
        int maturity = randIntInRange(648, 729);
        RandOption o(maturity, 12, 0.1, true);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    while (getApproxNumOfOperations(currentTotalWidth, currentTotalHeight) < constProduct)
    {
        int maturity = randIntInRange(1, 81);
        real reversionRate = getRandRRByWidthRangeTscAndRRRange(12, 0.01, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalHeight, currentTotalWidth);
    }

    string filename = "6_SKEWEDCONSTWIDTH";
    writeOptionsToFile(options, filename, 6, constProduct, skewPerc);
    cout << "finished writing to " << filename << endl;
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
    long constProduct = getApproxNumOfOperations(totalWidth, totalHeight);

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