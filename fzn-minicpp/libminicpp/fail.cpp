#include "fail.hpp"

#if !defined(CPPFAIL)
__thread jmp_buf* ptr = 0;
#endif
