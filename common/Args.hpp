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

    Args(int argc, char *argv[])
    {
        GetOpt_pp cmd(argc, argv);
        string s;
        
        version = 1;
        sort = SortType::NONE;

        cmd >> GetOpt::Option('o', "options", options);
        cmd >> GetOpt::Option('y', "yield", yield);
        cmd >> GetOpt::Option('s', "sort", s);
        cmd >> GetOpt::Option('v', "version", version);
        cmd >> GetOpt::OptionPresent('t', "test", test);

        if (strcasecmp(s.c_str(), string(1, (char)SortType::WIDTH).c_str()) == 0)
        {
            sort = SortType::WIDTH;
        }
        else if (strcasecmp(s.c_str(), string(1, (char)SortType::HEIGHT).c_str()) == 0)
        {
            sort = SortType::HEIGHT;
        }
    }
};
}

#endif
