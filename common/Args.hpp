#ifndef ARGS_HPP
#define ARGS_HPP

#include "getoptpp/getopt_pp_standalone.h"

using namespace std;
using namespace GetOpt;

namespace trinom
{

struct Args
{
  public:
    string options;
    string yield;
    bool test;

    static Args parseArgs(int argc, char *argv[])
    {
        GetOpt_pp cmd(argc, argv);
        Args args;

        cmd >> GetOpt::Option('o', "options", args.options, "");
        cmd >> GetOpt::Option('y', "yield", args.yield, "");
        cmd >> GetOpt::OptionPresent('t', "test", args.test);

        return args;
    }
};
}

#endif
