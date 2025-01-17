#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include "Types.h"

namespace Fzn
{
    class Var
    {
        public:
            domain_t domain;
            std::vector<annotation_t> annotations;
    };

    class ArrayVar
    {
        public:
            std::vector<identifier_t> variables;
            std::vector<annotation_t> annotations;
    };

    class Constraint
    {
        public:
            pred_identifier_t identifier;
            std::vector<constraint_arg_t> arguments;
            std::vector<annotation_t> annotations;
    };

    class Model
    {
        public:
            // FlatZinc
            std::vector<char> fzn_file_content;
            // Identifier -> Variable
            std::unordered_map<std::string, Var> int_vars;
            std::unordered_map<std::string, Var> bool_vars;
            // Constraints
            std::vector<Constraint> constraints;
            // Identifier -> Array of constants
            std::unordered_map<std::string, std::vector<int>> array_int_consts;
            std::unordered_map<std::string, std::vector<bool>> array_bool_consts;
            // Identifier -> Array of variables;
            std::unordered_map<std::string, ArrayVar> array_int_vars;
            std::unordered_map<std::string, ArrayVar> array_bool_vars;
            // Search strategy
            std::vector<search_annotation_t> search_strategy;
            // Objective variable
            identifier_t objective_var;
            // Solving type
            std::string_view solve_type;

        public:
            Model() = default;
    };
}
