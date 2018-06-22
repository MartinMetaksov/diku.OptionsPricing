#ifndef ARGS_HPP
#define ARGS_HPP

#include <string.h>
#include <vector>

#include "getoptpp/getopt_pp_standalone.h"

#include "OptionConstants.hpp"

using namespace GetOpt;

namespace trinom
{

struct Args
{
    std::string options;
    std::string yield;
    std::vector<SortType> sorts;
    std::vector<int> blockSizes;
    std::vector<int> versions;
    bool test;
    int runs;
    int device;

    Args() {}

    Args(int argc, char *argv[])
    {
        GetOpt_pp cmd(argc, argv);
        std::vector<std::string> sortOpts;

        // Defaults for single arguments
        runs = 0;
        device = 0;

        cmd >> GetOpt::Option('o', "options", options);
        cmd >> GetOpt::Option('y', "yield", yield);
        cmd >> GetOpt::Option('s', "sort", sortOpts);
        cmd >> GetOpt::Option('v', "version", versions);
        cmd >> GetOpt::Option('r', "runs", runs);
        cmd >> GetOpt::Option('b', "block", blockSizes);
        cmd >> GetOpt::Option('d', "device", device);
        cmd >> GetOpt::OptionPresent('t', "test", test);

        for (auto &sort : sortOpts)
        {
            if (sort.length() == 1)
            {
                auto sortType = (SortType)sort[0];
                if (sortType == SortType::HEIGHT_ASC || sortType == SortType::HEIGHT_DESC || sortType == SortType::WIDTH_ASC //
                    || sortType == SortType::WIDTH_DESC || sortType == SortType::NONE)
                {
                    sorts.push_back(sortType);
                }
            }
        }

        // Defaults for multiple arguments
        if (sorts.empty())
            sorts.push_back(SortType::NONE);
        if (blockSizes.empty())
            blockSizes.push_back(1024);
        if (versions.empty())
            versions.push_back(1);
    }
};
} // namespace trinom

#endif
