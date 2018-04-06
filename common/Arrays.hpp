#ifndef ARRAYS_HPP
#define ARRAYS_HPP

#include <type_traits>
#include <vector>
#include <iostream>
#include <limits>

using namespace std;

namespace trinom
{

class Arrays
{

  public:
	template <class T,
			  class = decltype(declval<istream &>() >> declval<T &>())>
	static void read_array(istream &in, vector<T> *array)
	{
		T x;
		char c;
		in >> c;
		while (!in.eof())
		{
			in >> x >> c;
			array->push_back(x);

			if (c == ']')
				break;
		}
	}

	template <class T,
			  class = decltype(declval<ostream &>() << declval<T>())>
	static void write_array(T *array, unsigned int length)
	{
		cout.precision(numeric_limits<T>::max_digits10);

		cout << '[' << fixed << array[0];
		for (unsigned int i = 1; i < length; ++i)
		{
			cout << ", " << fixed << array[i];
		}
		cout << ']' << endl;
	}
};
}

#endif