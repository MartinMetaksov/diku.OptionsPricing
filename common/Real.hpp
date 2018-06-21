#ifndef REAL_H
#define REAL_H

namespace trinom
{

// Define this to use doubles
#define USE_DOUBLE
#ifdef USE_DOUBLE
typedef double real;
#define ROUND(X) round((X))
#else
typedef float real;
#define ROUND(X) roundf((X))
#endif

#define zero (real)0.0
#define one (real)1.0
#define two (real)2.0
#define half (real)0.5
#define three (real)3.0
#define six (real)6.0
#define seven (real)7.0
#define year (real)365.0
#define minus184 (real)-0.184

} // namespace trinom

#endif
