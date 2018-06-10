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

#define zero 0.0
#define one 1.0
#define two 2.0
#define half 0.5
#define three 3.0
#define six 6.0
#define seven 7.0
#define year 365.0
#define minus184 -0.184

//const real zero = 0;
//const real one = 1;
//const real two = 2;
//const real half = one / two;
//const real three = 3;
//const real six = 6;
//const real seven = 7;
//const real year = 365;
//const real minus184 = -0.184;
} // namespace trinom

#endif
