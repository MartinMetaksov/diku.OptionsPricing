#ifndef REAL_H
#define REAL_H

#include <cfloat>

namespace trinom
{

// Define this to use doubles
// #define USE_DOUBLE
#ifdef USE_DOUBLE
typedef double real;
#define ROUND(X) round((X))
#define REAL_MAX DBL_MAX
#define REAL_MIN DBL_MIN
#else
typedef float real;
#define ROUND(X) roundf((X))
#define REAL_MAX FLT_MAX
#define REAL_MIN FLT_MIN
#endif

#define infinity (real) INFINITY
#define zero (real)0.0
#define one (real)1.0
#define two (real)2.0
#define half (real)0.5
#define three (real)3.0
#define six (real)6.0
#define seven (real)7.0
#define year (real)365.0
#define minus184 (real) - 0.184

} // namespace trinom

#endif
