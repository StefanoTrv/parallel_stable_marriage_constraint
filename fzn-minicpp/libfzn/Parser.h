#pragma once

#include <any>
#include <tuple>
#include <string>
#include <unordered_set>
#include <vector>

#include "Model.h"
#include "peglib/peglib.h"

namespace Fzn
{
    class Parser
    {
        private:
            peg::parser fzn_parser;

        public:
            Parser();
            Model parse(std::string const & fzn_file_path);
        private:
            static void read_fzn_file(std::string const & fzn_file_path, std::vector<char> & buffer);
            void add_constants_actions();
            void add_variables_actions();
            void add_constraints_actions();
            void add_literals_actions();
            void add_identifiers_actions();
            void add_ranges_sets_actions();
            void add_types_actions();
            void add_annotation_actions();
            void add_expressions_actions();
            void add_solve_actions();
            void add_search_annotations_actions();
            template<typename T>
            static std::vector<T> sequence_action(peg::SemanticValues const & vs);
    };
}


template<typename T>
std::vector<T> Fzn::Parser::sequence_action(peg::SemanticValues const & vs)
{
    auto const count = vs.size();
    std::vector<T> sequence(count);
    for (auto i = 0; i < count; i +=1)
    {
        sequence.at(i) = std::move(std::any_cast<T>(vs.at(i)));
    }
    return sequence;
}