#ifndef SEQ_HPP
#define SEQ_HPP

#include "../common/OptionConstants.hpp"
#include "../common/Domain.hpp"

using namespace trinom;

namespace seq
{

struct jvalue
{
    real rate;
    real pu;
    real pm;
    real pd;
};

/**
 *  Sequential version that computes the bond tree until bond maturity
 *  and prices the option on maturity during backward propagation.
 **/
real computeSingleOption(const OptionConstants &c, const Yield &yield)
{
    // Precompute probabilities and rates for all js.
    auto jvalues = new jvalue[c.width];
    auto jmin = -c.jmax;

    jvalue &valmin = jvalues[0];
    valmin.rate = jmin * c.dr;
    valmin.pu = PU_B(jmin, c.M);
    valmin.pm = PM_B(jmin, c.M);
    valmin.pd = PD_B(jmin, c.M);

    jvalue &valmax = jvalues[c.width - 1];
    valmax.rate = c.jmax * c.dr;
    valmax.pu = PU_C(c.jmax, c.M);
    valmax.pm = PM_C(c.jmax, c.M);
    valmax.pd = PD_C(c.jmax, c.M);

    for (auto i = 1; i < c.width - 1; ++i)
    {
        jvalue &val = jvalues[i];
        auto j = i + jmin;
        val.rate = j * c.dr;
        val.pu = PU_A(j, c.M);
        val.pm = PM_A(j, c.M);
        val.pd = PD_A(j, c.M);
    }

    // Forward induction to calculate Qs and alphas
    auto Qs = new real[c.width]();     // Qs[j]: j in jmin..jmax
    auto QsCopy = new real[c.width](); // QsCopy[j]
    Qs[c.jmax] = one;                  // Qs[0] = 1$

    auto alphas = new real[c.n + 1]();                              // alphas[i]
    alphas[0] = getYieldAtYear(c.dt, c.termUnit, yield.Prices.data(), yield.TimeSteps.data(), yield.N); // initial dt-period interest rate

    for (auto i = 0; i < c.n; ++i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = alphas[i];

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - jmin;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            auto qexp = Qs[jind] * exp(-(alpha + jval.rate) * c.dt);

            if (j == jmin)
            {
                // Bottom edge branching
                QsCopy[jind + 2] += jval.pu * qexp; // up two
                QsCopy[jind + 1] += jval.pm * qexp; // up one
                QsCopy[jind] += jval.pd * qexp;     // middle
            }
            else if (j == c.jmax)
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
        auto jhigh1 = min(i + 1, c.jmax);
        real alpha_val = 0;
        for (auto j = -jhigh1; j <= jhigh1; ++j)
        {
            auto jind = j - jmin;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            alpha_val += QsCopy[jind] * exp(-jval.rate * c.dt);
        }

        alphas[i + 1] = computeAlpha(alpha_val, i, c.dt, c.termUnit, yield.Prices.data(), yield.TimeSteps.data(), yield.N);

        // Switch Qs
        auto QsT = Qs;
        Qs = QsCopy;
        QsCopy = QsT;
        fill_n(QsCopy, c.width, 0);
    }

    // Backward propagation
    auto call = Qs; // call[j]
    auto callCopy = QsCopy;

    fill_n(call, c.width, 100); // initialize to 100$

    for (auto i = c.n - 1; i >= 0; --i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = alphas[i];
        auto isMaturity = i == ((int)(c.t / c.dt));

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - jmin;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            auto callExp = exp(-(alpha + jval.rate) * c.dt);

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (jval.pu * call[jind] +
                       jval.pm * call[jind - 1] +
                       jval.pd * call[jind - 2]) *
                      callExp;
            }
            else if (j == jmin)
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
            callCopy[jind] = computeCallValue(isMaturity, c, res);
        }

        // Switch call arrays
        auto callT = call;
        call = callCopy;
        callCopy = callT;

        fill_n(callCopy, c.width, 0);
    }

    auto result = call[c.jmax];

    delete[] jvalues;
    delete[] alphas;
    delete[] Qs;
    delete[] QsCopy;

    return result;
}

void computeOptions(const Options &options, const Yield &yield, vector<real> &results)
{
    for (auto i = 0; i < options.N; ++i)
    {
        OptionConstants c(options, i);
        auto result = computeSingleOption(c, yield);
        results.push_back(result);
    }
}

}

#endif
