
// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#include "CudaOption.cuh"
#include "../common/FutharkArrays.hpp"

using namespace std;

void computeAllOptions(const string &filename, bool isTest)
{
    // Read options from filename, allocate the result array
    auto options = Option::read_options(filename);
    auto length = options.size();
    auto optionConstants = new OptionConstants[length];

    for (auto i = 0; i < length; ++i)
    {
        optionConstants[i] = OptionConstants::computeConstants(options.at(i));
    }

    auto result = new real[length];
    computeCuda(optionConstants, result, length, isTest);

    // Don't print mock results
    FutharkArrays::write_futhark_array(result, length);

    delete[] result;
    delete[] optionConstants;
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

    computeAllOptions(filename, isTest);

    return 0;
}
