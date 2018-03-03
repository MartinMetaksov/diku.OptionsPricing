#include "../common/Domain.hpp"
#include "../common/Option.hpp"
#include "../common/FutharkArrays.hpp"

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

    //----------------------
    // Compute Q values
    //-----------------------
    // Define initial tree values
    auto m = jmax + 2;
    auto Qlen = 2 * m + 1;
    auto Q = new real[Qlen]();
    Q[m] = one;

    auto alphas = new real[n + 1](); // [n+1]real
    alphas[0] = h_YieldCurve[0].p;

    auto QCopy = new real[Qlen]();

    // forward propagation loop
    for (auto i = 0; i < n; ++i)
    {
        auto imax = min(i + 1, jmax);

        // Reset
        memcpy(QCopy, Q, sizeof(real) * Qlen);

        //--------------------------
        // forward iteration step --
        //--------------------------
        for (auto jj = 0; jj < Qlen; ++jj)
        {
            auto j = jj - m;
            if (j < -imax || j > imax)
            {
                Q[jj] = zero;
            }
            else
            {
                Q[jj] = fwdHelper(M, dr, dt, alphas[i], QCopy, 0, m, i, imax, jmax, j);
            }
        }

        // determine new alphas
        real alpha_val = 0;
        for (auto jj = 0; jj < Qlen; ++jj)
        {
            auto j = jj - imax;
            alpha_val += (j < -imax || j > imax) ? 0 : Q[j + m] * exp(-((real)j) * dr * dt);
        }

        // interpolation of yield curve
        real t = (i + 1) * dt;
        auto R = getYieldAtDay(t * year);

        alphas[i + 1] = log(alpha_val / exp(-R * t));
    }

    //---------------------------------------------------------
    // Compute values at expiration date:
    // call option value at period end is V(T) = S(T) - X
    // if S(T) is greater than X, or zero otherwise.
    // The computation is similar for put options.
    //---------------------------------------------------------
    auto Call = Q;
    for (auto j = 0; j < Qlen; ++j)
    {
        Call[j] = ((j >= -jmax + m) && (j <= jmax + m)) ? one : zero;
    }

    auto CallCopy = QCopy;

    // back propagation loop
    for (auto ii = 0; ii < n; ++ii)
    {
        auto i = n - 1 - ii;
        auto imax = min(i + 1, jmax);

        // Copy array values to avoid overwriting during update
        memcpy(CallCopy, Call, sizeof(real) * Qlen);

        //---------------------------
        // backward iteration step --
        //---------------------------
        for (auto jj = 0; jj < Qlen; ++jj)
        {
            auto j = jj - m;
            if (j < -imax || j > imax)
            {
                Call[jj] = zero;
            }
            else
            {
                Call[jj] = bkwdHelper(X, M, dr, dt, alphas[i], CallCopy, 0, m, i, jmax, j);
            }
        }
    }

    auto ret = Call[m];
    delete[] Q;
    delete[] QCopy;
    delete[] alphas;

    return ret;
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
