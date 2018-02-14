#ifndef REAL_H
#define REAL_H

// Define this to use doubles
//#define USE_DOUBLE
#ifdef USE_DOUBLE
typedef double real;
#else
typedef float real;
#endif

#endif
