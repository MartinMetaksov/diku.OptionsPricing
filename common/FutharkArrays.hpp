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
			cin >> x >> c >> t >> c;
			array->push_back(x);

			if (c == ']')
				break;
		}
	}

	template <class T>
	typename std::enable_if<std::is_arithmetic<T>::value, void>::type static write_futhark_array(T *array, unsigned int length)
	{
		cout.precision(numeric_limits<T>::max_digits10);
		string type;
		if (is_same<T, int>::value)
		{
			type = "i32";
		}
		else if (is_same<T, float>::value)
		{
			type = "f32";
		}
		else if (is_same<T, double>::value)
		{
			type = "f64";
		}
		else
		{
			type = "";
		}

		cout << '[' << fixed << array[0] << type;
		for (unsigned int i = 1; i < length; ++i)
		{
			cout << ", " << fixed << array[i] << type;
		}
		cout << ']' << endl;
	}
};

#endif