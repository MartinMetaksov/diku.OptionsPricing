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

/**
 *  Sequential version that computes the bond tree until option maturity
 *  and then analytically computes the option payoff.
**/
real compute_single_option(const Option &option)
{
    const real T = option.Maturity;
    const real t = option.Length;
    const int termUnitsInYearCount = ceil((real)year / option.TermUnit);
    const int n = option.TermStepCount * termUnitsInYearCount * t;
    const real dt = termUnitsInYearCount / (real)option.TermStepCount; // [years]

    const real X = option.StrikePrice;
    const real a = option.ReversionRate;
    const real sigma = option.Volatility;
    const real V = sigma * sigma * (one - exp(-two * a * dt)) / (two * a);
    const real dr = sqrt(three * V);
    const real M = exp(-a * dt) - one;

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
    auto Qs = new real[width]();     // Qs[j]: j in jmin..jmax
    auto QsCopy = new real[width](); // QsCopy[j]
    Qs[jmax] = one;                  // Qs[0] = 1$

    auto alphas = new real[n + 1](); // alphas[i]
    alphas[0] = getYieldAtYear(dt);  // initial dt-period interest rate

    real P_length = 0;
    real P_length1 = 0;

    for (auto i = 0; i < n; ++i)
    {
        auto jhigh = min(i, jmax);
        auto alpha = alphas[i];

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - jmin;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            auto qexp = Qs[jind] * exp(-(alpha + jval.rate) * dt);

            if (j - 1 < jmin)
            {
                // Bottom edge branching
                QsCopy[jind + 2] += jval.pu * qexp; // up two
                QsCopy[jind + 1] += jval.pm * qexp; // up one
                QsCopy[jind] += jval.pd * qexp;     // middle
            }
            else if (j + 1 > jmax)
            {
                // Top edge branching
                QsCopy[jind] += jval.pu * qexp;     // middle
                QsCopy[jind - 1] += jval.pm * qexp; // down one
                QsCopy[jind - 2] += jval.pd * qexp; // down two
            }
            else
            {
                // Standard branching
                QsCopy[jind + 1] += jval.pu * qexp; // up
                QsCopy[jind] += jval.pm * qexp;     // middle
                QsCopy[jind - 1] += jval.pd * qexp; // down
            }
        }

        // Determine the new alpha using equation 30.22
        // by summing up Qs from the next time step
        auto jhigh1 = min(i + 1, jmax);
        real alpha_val = 0;
        for (auto j = -jhigh1; j <= jhigh1; ++j)
        {
            auto jind = j - jmin;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            alpha_val += QsCopy[jind] * exp(-jval.rate * dt);
        }

        auto ti = (i + 2) * dt;             // next next time step
        auto R = getYieldAtYear(ti);        // discount rate
        auto P = exp(-R * ti);              // discount bond price
        alphas[i + 1] = log(alpha_val / P); // new alpha

        // Switch Qs
        auto QsT = Qs;
        Qs = QsCopy;
        QsCopy = QsT;
        fill_n(QsCopy, width, 0);

        if (i == n - 2)
        {
            P_length = P;
        }
        else if (i == n - 1)
        {
            P_length1 = P;
        }
    }

    // Backward propagation
    auto call = Qs; // call[j]
    auto callCopy = QsCopy;

    auto F_length = 0.07830417; //TODO: compute somehow
    auto R_maturity = getYieldAtYear(T);
    auto P_maturity = exp(-R_maturity * T);
    auto B_maturity = (1 - exp(-a * (T - t))) / a;
    auto B_length = (1 - exp(-a)) / a;
    auto Vt = sigma * sigma * (1 - exp(-2 * a * t)) / (4 * a);
    auto A_maturity = P_maturity / P_length * exp(B_maturity * F_length - (Vt * B_maturity * B_maturity));
    auto A_length = P_length1 / P_length * exp(B_length * F_length - (Vt * B_length * B_length));
    auto A_length_log = log(A_length);

    for (auto j = jmin; j <= jmax; ++j)
    {
        auto jind = j - jmin;                        // array index for j
        auto jval = jvalues[jind];                   // precomputed probabilities and rates
        auto R = alphas[n] + jval.rate;              // dt-period rate
        auto r = (R * dt + A_length_log) / B_length; // instantaneous rate
        real P = A_maturity * exp(-B_maturity * r);  // bond price
        auto payoff = max(X - 100 * P, zero);
        call[jind] = payoff;
    }

    for (auto i = n - 1; i >= 0; --i)
    {
        auto jhigh = min(i, jmax);
        auto alpha = alphas[i];

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - jmin;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            auto callExp = exp(-(alpha + jval.rate) * dt);

            real res;
            if (j == jmax)
            {
                // Top edge branching
                res = (jval.pu * call[jind] +
                       jval.pm * call[jind - 1] +
                       jval.pd * call[jind - 2]) *
                      callExp;
            }
            else if (j == -jmax)
            {
                // Bottom edge branching
                res = (jval.pu * call[jind + 2] +
                       jval.pm * call[jind + 1] +
                       jval.pd * call[jind]) *
                      callExp;
            }
            else
            {
                // Standard branching
                res = (jval.pu * call[jind + 1] +
                       jval.pm * call[jind] +
                       jval.pd * call[jind - 1]) *
                      callExp;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            callCopy[jind] = res;
        }

        // Switch call arrays
        auto callT = call;
        call = callCopy;
        callCopy = callT;
    }

    auto result = call[jmax];

    delete[] jvalues;
    delete[] alphas;
    delete[] Qs;
    delete[] QsCopy;

    return result;
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
