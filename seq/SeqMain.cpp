#include "../common/Domain.hpp"
#include "../common/Option.hpp"

int main(int argc, char *argv[])
{
    // Read options from stdin, allocate the result array
    vector<Option> options = Option::read_options();
    auto result = new real[options.size()];

    bool isTest = false;
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp(argv[i], "-test") == 0)
        {
            isTest = true;
        }
    }

    return 0;
}