#ifndef FUTHARK_ARRAYS_HPP
#define FUTHARK_ARRAYS_HPP

#include <type_traits>
#include <vector>
#include <iostream>
#include <limits>

using namespace std;

class FutharkArrays
{
  public:
	template <class T>
	typename std::enable_if<std::is_arithmetic<T>::value, void>::type static read_futhark_array(vector<T> *array)
	{
		T x;
		int t;
		char c;
		cin >> c;
		while (!cin.eof())
		{
			cin >> x >> c;
			array->push_back(x);

			if (c == ']')
				break;
		}
	}

	template <class T>
	typename std::enable_if<std::is_arithmetic<T>::value, void>::type static write_futhark_array(T *array, unsigned int length)
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

#endif