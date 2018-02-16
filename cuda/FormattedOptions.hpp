#ifndef FORMATTED_OPTIONS_HPP
#define FORMATTED_OPTIONS_HPP

#include <vector>
#include <cmath>
#include "../common/Real.hpp"
#include "../common/Option.hpp"

using namespace std;

struct FormattedOptions
{
  public:
    int w;
    int n_max;
    int max_options_in_chunk;
    vector<int> options_in_chunk;
    vector<int> option_indices;
    vector<Option> options;

    static FormattedOptions format_options(int w, vector<Option> options)
    {
        int n_max = 0, m_max = 0;
        for (Option option : options)
        {
            auto T = option.maturity;
            auto n = option.num_of_terms;
            auto dt = T / ((real)n);
            auto a = option.reversion_rate;
            auto M = (exp(0 - a * dt)) - 1;
            auto jmax = (int)(-0.184 / M) + 1;
            auto m = jmax + 2;
            auto np1 = n + 1;
            auto m2p1 = 2 * m + 1;

            if (np1 > n_max)
            {
                n_max = np1;
            }
            if (m2p1 > m_max)
            {
                m_max = m2p1;
            }
        }

        int num_options = options.size();
        int max_options_in_chunk = w / (m_max + 1);
        int num_chunks = (num_options + max_options_in_chunk - 1) / max_options_in_chunk;

        FormattedOptions foptions;
        foptions.w = w;
        foptions.n_max = n_max;
        foptions.max_options_in_chunk = max_options_in_chunk;
        foptions.options_in_chunk.reserve(num_chunks);
        foptions.option_indices.reserve(num_chunks * max_options_in_chunk);
        foptions.options = options;

        for (int c_ind = 0; c_ind < num_chunks; ++c_ind)
        {
            int num = (c_ind == num_chunks - 1) ? (num_options - c_ind * max_options_in_chunk) : max_options_in_chunk;
            foptions.options_in_chunk.push_back(num);

            for (int i = 0; i < max_options_in_chunk; ++i)
            {
                int opt_ind = c_ind * max_options_in_chunk + i;
                int ind = (opt_ind < num_options) ? opt_ind : -1;
                foptions.option_indices.push_back(ind);
            }
        }
        return foptions;
    }
};

#endif
