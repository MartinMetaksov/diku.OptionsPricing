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

void print_array_csv(ostream &out, real *array, int size)
{
    for (auto i = 0; i < size; ++i)
    {
        out << array[i] << ',';
    }
}

/**
 *  Sequential version that computes the bond tree until bond maturity
 *  and prices the option on maturity during backward propagation.
**/
real compute_single_option(const Option &option)
{
    auto c = computeConstants(option);

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

    auto alphas = new real[c.n + 1](); // alphas[i]
    alphas[0] = getYieldAtYear(c.dt);  // initial dt-period interest rate

    ofstream out("forward-" + to_string(c.n) + ".csv");

    out << "i,alpha,";
    for (auto j = jmin; j <= c.jmax; ++j)
    {
        out << "Qs[" << j << "],";
    }
    out << endl;

    for (auto i = 0; i < c.n; ++i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = alphas[i];
        out << i << ',' << alpha << ',';
        print_array_csv(out, Qs, c.width);
        out << endl;

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - jmin;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            auto qexp = Qs[jind] * exp(-(alpha + jval.rate) * c.dt);

            if (j - 1 < jmin)
            {
                // Bottom edge branching
                QsCopy[jind + 2] += jval.pu * qexp; // up two
                QsCopy[jind + 1] += jval.pm * qexp; // up one
                QsCopy[jind] += jval.pd * qexp;     // middle
            }
            else if (j + 1 > c.jmax)
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

        auto ti = (i + 2) * c.dt;           // next next time step
        auto R = getYieldAtYear(ti);        // discount rate
        auto P = exp(-R * ti);              // discount bond price
        alphas[i + 1] = log(alpha_val / P); // new alpha

        // Switch Qs
        auto QsT = Qs;
        Qs = QsCopy;
        QsCopy = QsT;
        fill_n(QsCopy, c.width, 0);
    }
    out << c.n << ',' << alphas[c.n] << ',';
    print_array_csv(out, Qs, c.width);
    out << endl;
    out.close();

    // Backward propagation
    auto call = Qs; // call[j]
    auto callCopy = QsCopy;

    fill_n(call, c.width, 100); // initialize to 100$

    ofstream out2("backward-" + to_string(c.n) + ".csv");

    out2 << "i,";
    for (auto j = jmin; j <= c.jmax; ++j)
    {
        out2 << "call[" << j << "],";
    }
    out2 << endl;

    out2 << c.n << ',';
    print_array_csv(out2, call, c.width);
    out2 << endl;

    for (auto i = c.n - 1; i >= 0; --i)
    {
        auto jhigh = min(i, c.jmax);
        auto alpha = alphas[i];
        auto isMaturity = i == ((int)(option.Length / c.dt));

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
            callCopy[jind] = isMaturity ? max(c.X - res, zero) : res;
        }

        out2 << i << ',';
        print_array_csv(out2, callCopy, c.width);
        out2 << endl;

        // Switch call arrays
        auto callT = call;
        call = callCopy;
        callCopy = callT;
    }

    out2.close();

    auto result = call[c.jmax];

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
