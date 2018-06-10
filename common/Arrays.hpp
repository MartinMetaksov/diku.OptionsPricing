#ifndef ARRAYS_HPP
#define ARRAYS_HPP

#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

namespace trinom
{

struct Arrays
{

    template <class T,
              class = decltype(std::declval<std::istream &>() >> std::declval<T &>())>
    static void read_array(std::istream &in, std::vector<T> &array)
    {
        T x;
        char c;
        in >> c;
        while (!in.eof())
        {
            in >> x >> c;
            array.push_back(x);

            if (c == ']')
            {
                break;
            }
        }
    }

    template <class T,
              class = decltype(std::declval<std::ostream &>() << std::declval<T>())>
    static void write_array(std::ostream &out, const std::vector<T> &array)
    {
        out.precision(std::numeric_limits<T>::max_digits10);

        out << '[' << std::fixed << array[0];
        for (unsigned int i = 1; i < array.size(); ++i)
        {
            out << ", " << std::fixed << array[i];
        }
        out << ']' << std::endl;
    }
};
} // namespace trinom

#endif