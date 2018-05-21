#ifndef ARGS_HPP
#define ARGS_HPP

#include <string.h>
#include "getoptpp/getopt_pp_standalone.h"
#include "OptionConstants.hpp"

using namespace std;
using namespace GetOpt;

namespace trinom
{

struct Args
{
    string options;
    string yield;
    SortType sort;
    bool test;
    int version;
    int runs;
    int blockSize;

    Args(int argc, char *argv[])
    {
        GetOpt_pp cmd(argc, argv);
        string s;

        version = 1;
        runs = 0;
        sort = SortType::NONE;

        cmd >> GetOpt::Option('o', "options", options);
        cmd >> GetOpt::Option('y', "yield", yield);
        cmd >> GetOpt::Option('s', "sort", s);
        cmd >> GetOpt::Option('v', "version", version);
        cmd >> GetOpt::Option('r', "runs", runs);
        cmd >> GetOpt::Option('b', "block", blockSize);
        cmd >> GetOpt::OptionPresent('t', "test", test);

        if (s.length() == 1)
        {
            auto sortType = (SortType)s[0];
            if (sortType == SortType::HEIGHT_ASC || sortType == SortType::HEIGHT_DESC || sortType == SortType::WIDTH_ASC || sortType == SortType::WIDTH_DESC)
            {
                sort = sortType;
            }
        }
    }
};
} // namespace trinom

#endif
