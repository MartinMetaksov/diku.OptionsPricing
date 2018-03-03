#include "../common/Domain.hpp"
#include "../common/Option.hpp"
#include "../common/FutharkArrays.hpp"

struct jvalue
{
    real rate;
    real pu;
    real pm;
    real pd;
};

real compute_single_option(const Option &option)
{
    auto X = option.strike_price;
    auto n = option.num_of_terms;
    auto dt = option.length / (real)option.num_of_terms;
    auto a = option.reversion_rate;
    auto sigma = option.volatility;
    auto V = sigma * sigma * (one - exp(-two * a * dt)) / (two * a);
    auto dr = sqrt(three * V);
    auto M = exp(-a * dt) - one;
    auto jmax = (int)(minus184 / M) + 1;
    auto jmin = -jmax;

    auto width = 2 * jmax + 1;

    // Compute Table 2.
    auto jvalues = new jvalue[width];

    jvalue &valmin = jvalues[0];
    valmin.rate = jmin * dr;
    valmin.pu = PU_B(jmin, M);
    valmin.pm = PM_B(jmin, M);
    valmin.pd = PD_B(jmin, M);

    jvalue &valmax = jvalues[width - 1];
    valmax.rate = jmax * dr;
    valmax.pu = PU_C(jmax, M);
    valmax.pm = PM_C(jmax, M);
    valmax.pd = PD_C(jmax, M);

    for (auto i = 1; i < width - 1; ++i)
    {
        jvalue &val = jvalues[i];
        auto j = i + jmin;
        val.rate = j * dr;
        val.pu = PU_A(j, M);
        val.pm = PM_A(j, M);
        val.pd = PD_A(j, M);
    }

    return 0;
}

void compute_all_options(const string &filename)
{
    // Read options from filename, allocate the result array
    auto options = Option::read_options(filename);
    auto result = new real[options.size()];

    for (int i = 0; i < options.size(); ++i)
    {
        result[i] = compute_single_option(options.at(i));
    }

    FutharkArrays::write_futhark_array(result, options.size());

    delete[] result;
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
