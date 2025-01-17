#pragma once

#include <any>
#include <cxxabi.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace Fzn
{
    template<typename T>
    bool isType(std::any const & a);

    std::string getTypeName(std::type_info const & ti);
}

template<typename T>
bool Fzn::isType(std::any const & a)
{
    return a.type() == typeid(T);
}

inline
std::string Fzn::getTypeName(std::type_info const & ti)
{
    using namespace std;

    int status = 0;
    char * demangled = abi::__cxa_demangle(ti.name(), nullptr, nullptr, &status);
    if (status == 0)
    {
        string type_name(demangled);
        free(demangled);
        return type_name;
    }
    else
    {
        throw runtime_error("Demangle failed");
    }
}