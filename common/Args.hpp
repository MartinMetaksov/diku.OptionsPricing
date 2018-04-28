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
  public:
    string options;
    string yield;
    SortType sort;
    bool test;
    int version;

    static Args parseArgs(int argc, char *argv[])
    {
        GetOpt_pp cmd(argc, argv);
        Args args;
        string sort;
        
        args.version = 1;
        args.sort = SortType::NONE;

        cmd >> GetOpt::Option('o', "options", args.options);
        cmd >> GetOpt::Option('y', "yield", args.yield);
        cmd >> GetOpt::Option('s', "sort", sort);
        cmd >> GetOpt::Option('v', "version", args.version);
        cmd >> GetOpt::OptionPresent('t', "test", args.test);

        if (strcasecmp(sort.c_str(), string(1, (char)SortType::WIDTH).c_str()) == 0)
        {
            args.sort = SortType::WIDTH;
        }
        else if (strcasecmp(sort.c_str(), string(1, (char)SortType::HEIGHT).c_str()) == 0)
        {
            args.sort = SortType::HEIGHT;
        }

        return args;
    }
};
}

#endif
