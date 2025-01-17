#pragma once

#include <cmath>

inline
double division(int numerator, int denominator)
{
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

inline
int floorDivision (int numerator, int denominator)
{
    return static_cast<int>(std::floor(division(numerator, denominator)));
}

inline
int ceilDivision (int numerator, int denominator)
{
    return static_cast<int>(std::ceil(division(numerator, denominator)));
}