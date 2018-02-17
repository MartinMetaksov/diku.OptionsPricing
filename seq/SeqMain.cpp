#include "../common/Domain.hpp"
#include "../common/Option.hpp"
#include "../common/FutharkArrays.hpp"

real compute_single_option(const Option &option)
{
    auto X = option.strike_price;
    auto T = option.maturity;
    auto n = option.num_of_terms;
    auto dt = T / ((real)n);
    auto a = option.reversion_rate;
    auto sigma = option.volatility;
    auto V = sigma * sigma * (one - (exp(zero - two * a * dt))) / (two * a);
    auto dr = sqrt((one + two) * V);
    auto M = (exp(zero - a * dt)) - one;
    auto jmax = (int)(-0.184 / M) + 1;
    auto m = jmax + 2;

    //----------------------
    // Compute Q values
    //-----------------------
    // Define initial tree values
    auto Qlen = 2 * m + 1;
    auto Q = new real[Qlen]();
    Q[m] = one;

    auto alphas = new real[n + 1](); // [n+1]real
    alphas[0] = h_YieldCurve[0].p;

    // TODO: forward propagation loop

    return 0;
}

void compute_all_options(const string &filename)
{
    // Read options from filename, allocate the result array
    vector<Option> options = Option::read_options(filename);
    auto result = new real[options.size()];

    for (int i = 0; i < options.size(); ++i)
    {
        result[i] = compute_single_option(options.at(i));
    }

    FutharkArrays::write_futhark_array(result, options.size());
}

int main(int argc, char *argv[])
{
    bool isTest = false;
    string filename;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-test") == 0)
        {
            isTest = true;
        }
        else
        {
            filename = argv[i];
        }
    }

    compute_all_options(filename);

    return 0;
}
