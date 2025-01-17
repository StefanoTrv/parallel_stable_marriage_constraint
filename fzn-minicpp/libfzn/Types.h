#pragma once

#include <string_view>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

namespace Fzn
{
    // Identifiers
    using identifier_t = std::string_view;
    using pred_identifier_t = std::string_view;
    // Constraints
    using constraint_arg_t = std::variant<std::monostate, bool, int, std::vector<bool>, std::vector<int>, identifier_t, std::vector<identifier_t>>;
    // Ranges and Sets
    using int_range_t = std::pair<int, int>;
    using int_set_t = std::vector<int>;
    using bool_set_t = std::vector<bool>;
    // Types
    using basic_literal_type_t = std::string_view;
    using array_literal_type_t = std::pair<int_range_t, basic_literal_type_t>;
    using basic_var_type_t = std::variant<basic_literal_type_t, int_range_t, int_set_t>;
    using array_var_type_t = std::pair<int_range_t, basic_var_type_t>;
    using solve_type_t = std::string_view;
    // Domains
    using domain_t = std::variant<bool_set_t,int_range_t,int_set_t>;
    // Expressions
    using array_literal_expr_t = std::variant<std::monostate, std::vector<bool>, std::vector<int>>;
    using fixed_var_expr_t = std::pair<std::string_view,std::variant<bool,int>>;
    using basic_var_expr_t = std::string_view;
    using var_expr_t = std::variant<basic_var_expr_t, std::vector<basic_var_expr_t>>;
    // Annotations
    using annotation_arg_t = std::variant<int, float, identifier_t, std::vector<int_range_t>>;
    using annotation_t = std::pair<pred_identifier_t, std::vector<annotation_arg_t>>;
    // Search annotations
    using basic_search_annotation_t = std::tuple<pred_identifier_t, var_expr_t, std::vector<annotation_t>>;
    using array_search_annotation_t = std::pair<pred_identifier_t, std::vector<basic_search_annotation_t>>;
    using search_annotation_t = std::variant<basic_search_annotation_t,array_search_annotation_t>;
}