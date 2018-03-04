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

    // simplified computations
    // dr = sigma * sqrt(three * dt);
    // M = -a * dt;

    auto jmax = (int)(minus184 / M) + 1;
    auto jmin = -jmax;
    auto width = 2 * jmax + 1;

    // Precompute probabilities and rates for all js.
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

    // Forward induction to calculate Qs and alphas
    // i in 0..n    j in jmin..jmax
    auto Qs = new real[(n + 1) * width](); // Qs[i][j]
    Qs[jmax] = one;                        // Qs[0][0] = 1$

    auto alphas = new real[n + 1]();      // alphas[i]
    alphas[0] = getYieldAtDay(dt * year); // initial dt-period interest rate

    for (auto i = 0; i < n; ++i)
    {
        auto jhigh = min(i, jmax);

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - jmin;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            auto qexp = Qs[i * width + jind] * exp(-(alphas[i] + jval.rate) * dt);

            if (j - 1 < jmin)
            {
                // Bottom edge branching
                Qs[(i + 1) * width + jind + 2] += jval.pu * qexp; // up two
                Qs[(i + 1) * width + jind + 1] += jval.pm * qexp; // up one
                Qs[(i + 1) * width + jind] += jval.pd * qexp;     // middle
            }
            else if (j + 1 > jmax)
            {
                // Top edge branching
                Qs[(i + 1) * width + jind] += jval.pu * qexp;     // middle
                Qs[(i + 1) * width + jind - 1] += jval.pm * qexp; // down one
                Qs[(i + 1) * width + jind - 2] += jval.pd * qexp; // down two
            }
            else
            {
                // Standard branching
                Qs[(i + 1) * width + jind + 1] += jval.pu * qexp; // up
                Qs[(i + 1) * width + jind] += jval.pm * qexp;     // middle
                Qs[(i + 1) * width + jind - 1] += jval.pd * qexp; // down
            }
        }

        // Determine the new alpha using equation 30.22
        // by summing up Qs from the next time step
        auto jhigh1 = min(i + 1, jmax);
        real alpha_val = 0;
        for (auto j = -jhigh1; j <= jhigh1; ++j)
        {
            auto jind = j - jmin; // array index for j
            alpha_val += Qs[(i + 1) * width + jind] * exp(-j * dr * dt);
        }

        auto t = (i + 2) * dt;            // next next time step
        auto R = getYieldAtDay(t * year); // discount rate
        auto P = exp(-R * t);             // discount bond price
        auto alpha = log(alpha_val / P);  // new alpha
        alphas[i + 1] = alpha;
    }

    delete[] Qs;
    delete[] jvalues;
    delete[] alphas;

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
