#include "../common/Domain.hpp"
#include "../common/Option.hpp"

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

    // Read options from filename, allocate the result array
    vector<Option> options = Option::read_options(filename);
    auto result = new real[options.size()];
    cout << options.size() << endl;

    return 0;
}