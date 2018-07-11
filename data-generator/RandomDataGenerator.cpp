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

real getRandRRByTscAndRRRange(int termStepCount, real a, real b)
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

void addOption(vector<RandOption> &options, const RandOption o, long &currentTotalProd)
{
    currentTotalProd += (o.Height * o.Width);
    options.push_back(o);
}

int getNumSkewed(const int x, const RandOption y) { return y.Skewed ? x + 1 : x; }

int getNumSkewedOptions(vector<RandOption> &options)
{
    return accumulate(options.begin(), options.end(), 0, getNumSkewed);
}

void writeOptionsToFile(vector<RandOption> &randOptions,
                        const string filename,
                        const int dataType,
                        const long constProduct,
                        const int skewPercent,
                        const long finalProduct,
                        const long finalSkewProduct)
{
    string dataFile = "../data/" + filename + ".in";
    string markdownFile = "../data/dataInfo/" + filename + ".md";

    std::ofstream mdFile(markdownFile);

    mdFile << "Filename: " << dataFile << endl;
    mdFile << "Total size: " << randOptions.size() << endl;
    mdFile << "Constant max product: " << constProduct << endl;
    mdFile << "Current product: " << finalProduct << endl;
    mdFile << "Deviation: " << abs(constProduct - finalProduct) * 100 / (real)constProduct << "%" << endl;
    if (dataType == 4 || dataType == 5 || dataType == 6)
    {
        mdFile << "Skew: " << skewPercent << "%" << endl;
        mdFile << "Total skewed options: " << getNumSkewedOptions(randOptions) << endl;
        mdFile << "Skewed product: " << finalSkewProduct << endl;
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
    long currentTotalProd = 0;

    while (currentTotalProd < constProduct)
    {
        RandOption o(27, 12, 0.1, false);
        addOption(options, o, currentTotalProd);
    }

    string filename = "0_UNIFORM";
    writeOptionsToFile(options, filename, 0, constProduct, 0, currentTotalProd, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_1(vector<RandOption> &options, const long constProduct)
{
    long currentTotalProd = 0;

    while (currentTotalProd < constProduct)
    {
        int maturity = randIntInRange(1, 729);
        real reversionRate = getRandRRByTscAndRRRange(12, 0.00433, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalProd);
    }

    string filename = "1_RAND";
    writeOptionsToFile(options, filename, 1, constProduct, 0, currentTotalProd, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_2(vector<RandOption> &options, const long constProduct)
{
    long currentTotalProd = 0;

    while (currentTotalProd < constProduct)
    {
        real reversionRate = getRandRRByTscAndRRRange(12, 0.00433, 0.99);
        RandOption o(9, 12, reversionRate, false);
        addOption(options, o, currentTotalProd);
    }

    string filename = "2_RANDCONSTHEIGHT";
    writeOptionsToFile(options, filename, 2, constProduct, 0, currentTotalProd, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_3(vector<RandOption> &options, const long constProduct)
{
    long currentTotalProd = 0;

    while (currentTotalProd < constProduct)
    {
        int maturity = randIntInRange(1, 729);
        RandOption o(maturity, 12, 0.1, false);
        addOption(options, o, currentTotalProd);
    }
    string filename = "3_RANDCONSTWIDTH";
    writeOptionsToFile(options, filename, 3, constProduct, 0, currentTotalProd, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_4(vector<RandOption> &options, const long constProduct, const int skewPerc)
{
    long currentTotalProd = 0;

    while (currentTotalProd < (skewPerc / (real)100) * constProduct)
    {
        int maturity = randIntInRange(648, 729);
        real reversionRate = getRandRRByTscAndRRRange(12, 0.00433, 0.0075);
        RandOption o(maturity, 12, reversionRate, true);
        addOption(options, o, currentTotalProd);
    }

    const long currentTotalSkewProd = currentTotalProd;

    while (currentTotalProd < constProduct)
    {
        int maturity = randIntInRange(1, 81);
        real reversionRate = getRandRRByTscAndRRRange(12, 0.01, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalProd);
    }

    string filename = "4_SKEWED";
    writeOptionsToFile(options, filename, 4, constProduct, skewPerc, currentTotalProd, currentTotalSkewProd);
    cout << "finished writing to " << filename << endl;
}

void distribute_5(vector<RandOption> &options, const long constProduct, const int skewPerc)
{
    long currentTotalProd = 0;

    while (currentTotalProd < (skewPerc / (real)100) * constProduct)
    {
        real reversionRate = getRandRRByTscAndRRRange(12, 0.00433, 0.0045);
        RandOption o(9, 12, reversionRate, true);
        addOption(options, o, currentTotalProd);
    }

    const long currentTotalSkewProd = currentTotalProd;

    while (currentTotalProd < constProduct)
    {
        int maturity = randIntInRange(1, 81);
        real reversionRate = getRandRRByTscAndRRRange(12, 0.01, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalProd);
    }

    string filename = "5_SKEWEDCONSTHEIGHT";
    writeOptionsToFile(options, filename, 5, constProduct, skewPerc, currentTotalProd, currentTotalSkewProd);
    cout << "finished writing to " << filename << endl;
}

void distribute_6(vector<RandOption> &options, const long constProduct, const int skewPerc)
{
    long currentTotalProd = 0;

    while (currentTotalProd < (skewPerc / (real)100) * constProduct)
    {
        int maturity = randIntInRange(648, 729);
        RandOption o(maturity, 12, 0.1, true);
        addOption(options, o, currentTotalProd);
    }

    const long currentTotalSkewProd = currentTotalProd;

    while (currentTotalProd < constProduct)
    {
        int maturity = randIntInRange(1, 81);
        real reversionRate = getRandRRByTscAndRRRange(12, 0.01, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalProd);
    }

    string filename = "6_SKEWEDCONSTWIDTH";
    writeOptionsToFile(options, filename, 6, constProduct, skewPerc, currentTotalProd, currentTotalSkewProd);
    cout << "finished writing to " << filename << endl;
}

void distribute_7(vector<RandOption> &options, const long constProduct)
{
    long currentTotalProd = 0;

    while (currentTotalProd < constProduct)
    {
        int maturity = randIntInRange(1, 300);
        real reversionRate = getRandRRByTscAndRRRange(12, 0.018, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalProd);
    }

    string filename = "7_RAND_HEIGHT_IN_0-3500";
    writeOptionsToFile(options, filename, 7, constProduct, 0, currentTotalProd, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_8(vector<RandOption> &options, const long constProduct)
{
    long currentTotalProd = 0;

    while (currentTotalProd < constProduct)
    {
        int maturity = randIntInRange(1, 100);
        real reversionRate = getRandRRByTscAndRRRange(12, 0.018, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalProd);
    }

    string filename = "8_RAND_HEIGHT_IN_0-1200";
    writeOptionsToFile(options, filename, 8, constProduct, 0, currentTotalProd, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_9(vector<RandOption> &options, const long constProduct)
{
    long currentTotalProd = 0;

    while (currentTotalProd < constProduct)
    {
        int maturity = randIntInRange(4, 20);
        real reversionRate = getRandRRByTscAndRRRange(12, 0.018, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalProd);
    }

    string filename = "9_RAND_HEIGHT_IN_50-250";
    writeOptionsToFile(options, filename, 9, constProduct, 0, currentTotalProd, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_10(vector<RandOption> &options, const long constProduct)
{
    long currentTotalProd = 0;

    while (currentTotalProd < constProduct)
    {
        int maturity = randIntInRange(4, 45);
        real reversionRate = getRandRRByTscAndRRRange(12, 0.018, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalProd);
    }

    string filename = "10_RAND_HEIGHT_IN_50-500";
    writeOptionsToFile(options, filename, 10, constProduct, 0, currentTotalProd, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_11(vector<RandOption> &options, const long constProduct)
{
    long currentTotalProd = 0;

    while (currentTotalProd < constProduct)
    {
        int maturity = randIntInRange(8, 25);
        real reversionRate = getRandRRByTscAndRRRange(12, 0.018, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalProd);
    }

    string filename = "11_RAND_HEIGHT_IN_100-300";
    writeOptionsToFile(options, filename, 10, constProduct, 0, currentTotalProd, 0);
    cout << "finished writing to " << filename << endl;
}

void distribute_12(vector<RandOption> &options, const long constProduct)
{
    long currentTotalProd = 0;

    while (currentTotalProd < constProduct)
    {
        int maturity = randIntInRange(8, 60);
        real reversionRate = getRandRRByTscAndRRRange(12, 0.018, 0.99);
        RandOption o(maturity, 12, reversionRate, false);
        addOption(options, o, currentTotalProd);
    }

    string filename = "12_RAND_HEIGHT_IN_100-700";
    writeOptionsToFile(options, filename, 11, constProduct, 0, currentTotalProd, 0);
    cout << "finished writing to " << filename << endl;
}

int main(int argc, char *argv[])
{
    int dataType;
    long totalProd;
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
    cmd >> GetOpt::Option('p', "totalProd", totalProd);
    cmd >> GetOpt::Option('s', "skewPerc", skewPercent);

    vector<RandOption> randOptions;

    switch (dataType)
    {
    case 0:
        distribute_0(randOptions, totalProd);
        break;
    case 1:
        distribute_1(randOptions, totalProd);
        break;
    case 2:
        distribute_2(randOptions, totalProd);
        break;
    case 3:
        distribute_3(randOptions, totalProd);
        break;
    case 4:
        distribute_4(randOptions, totalProd, skewPercent);
        break;
    case 5:
        distribute_5(randOptions, totalProd, skewPercent);
        break;
    case 6:
        distribute_6(randOptions, totalProd, skewPercent);
        break;
    case 7:
        distribute_7(randOptions, totalProd);
        break;
    case 8:
        distribute_8(randOptions, totalProd);
        break;
    case 9:
        distribute_9(randOptions, totalProd);
        break;
    case 10:
        distribute_10(randOptions, totalProd);
        break;
    case 11:
        distribute_11(randOptions, totalProd);
        break;
    case 12:
        distribute_12(randOptions, totalProd);
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
        distribute_0(rand0, totalProd);
        distribute_1(rand1, totalProd);
        distribute_2(rand2, totalProd);
        distribute_3(rand3, totalProd);
        distribute_4(rand4, totalProd, skewPercent);
        distribute_5(rand5, totalProd, skewPercent);
        distribute_6(rand6, totalProd, skewPercent);
    }

    return 0;
}