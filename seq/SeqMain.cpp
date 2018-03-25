#include "../common/Domain.hpp"
#include "../common/OptionConstants.hpp"
#include "../common/FutharkArrays.hpp"
#include "Seq.hpp"
#include <cstring>

using namespace trinom;

void computeAllOptions(const string &filename)
{
    // Read options from filename, allocate the result array
    auto options = Option::read_options(filename);
    auto result = new real[options.size()];

    for (int i = 0; i < options.size(); ++i)
    {
        auto c = OptionConstants::computeConstants(options.at(i));
        result[i] = seq::computeSingleOption(c);
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

    computeAllOptions(filename);

    return 0;
}
