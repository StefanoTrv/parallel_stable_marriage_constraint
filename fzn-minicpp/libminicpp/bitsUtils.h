#pragma once

#include <climits>

inline
int getLeftmostOneIndex32(unsigned int val)
{
    return __builtin_clz(val);
}

inline
int getRightmostOneIndex64(unsigned long long int const & val)
{
    return 64 - __builtin_ffsll(val);
}

inline
int getRightmostOneIndex32(unsigned int val)
{
    return 32 - __builtin_ffs(val);
}

inline
unsigned int getMask32(int bitIndex)
{
    return 1 << (31 - bitIndex);
}

inline
unsigned int getLeftFilledMask32(int bitIndex)
{
    return UINT_MAX << (31 - bitIndex);
}

inline
unsigned int getRightFilledMask32(int bitIndex)
{
    return UINT_MAX >> bitIndex;
}


inline
int getPopCount(unsigned int val)
{
    return __builtin_popcount(val);
}