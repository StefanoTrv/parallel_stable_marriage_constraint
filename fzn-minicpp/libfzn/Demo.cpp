#include <cstdlib>
#include <iostream>
#include <string>

#include "Model.h"
#include "Parser.h"
#include "Printer.h"

using namespace std;

void add_outputs(Fzn::Model const & fzn_model, Fzn::Printer & fzn_printer);

int main(int argc, char *argv[])
{
    // Arguments check
    if (argc != 2)
    {
        cout << "Usage example: fzn-demo nqueens.fzn" << endl;
        exit(EXIT_FAILURE);
    }
    char const * fzn_file_path = argv[1];

    // Parsing the FlatZinc file
    Fzn::Parser fzn_parser;
    Fzn::Model const & fzn_model = fzn_parser.parse(fzn_file_path);

    // Print a solution with random values
    Fzn::Printer fzn_printer;
    add_outputs(fzn_model, fzn_printer);
    fzn_printer.print_outputs(cout);
    cout << "----------" << endl;

    // Print basic model statistics
    cout << "%%%mzn-stat: variables=" << fzn_model.int_vars.size() + fzn_model.bool_vars.size() << endl;
    cout << "%%%mzn-stat: propagators=" << fzn_model.constraints.size() << endl;
    cout << "%%%mzn-stat-end" << endl;
}

void add_outputs(Fzn::Model const & fzn_model, Fzn::Printer & fzn_printer)
{
    srand(time(nullptr));

    // Integer variables
    for (auto const & entry : fzn_model.int_vars)
    {
        auto const & identifier = entry.first;
        auto const & int_var = entry.second;
        for (auto const & annotation : int_var.annotations)
        {
            if (annotation.first == "output_var")
            {
                    auto callback = []() -> int {return rand() % 1000;};
                    fzn_printer.add_output<int>(identifier, callback);
            }
        }
    }

    // Bool variables
    for (auto const & entry : fzn_model.bool_vars)
    {
        auto const & identifier = entry.first;
        auto const & bool_var = entry.second;
        for (auto const & annotation : bool_var.annotations)
        {
            if (annotation.first == "output_var")
            {
                auto callback = []() -> int {return rand() % 2 == 0;};
                fzn_printer.add_output<int>(identifier, callback);
            }
        }
    }

    // Arrays of integer variables
    for (auto const & entry : fzn_model.array_int_vars)
    {
        auto const & identifier = entry.first;
        auto const & array_int_var = entry.second;
        for (auto const & annotation : array_int_var.annotations)
        {
            if (annotation.first == "output_array")
            {
                auto indices = get<vector<Fzn::int_range_t>>(annotation.second.at(0));
                auto callback = []() -> int {return rand() % 1000;};
                vector<function<int()>> callbacks(array_int_var.variables.size(),  callback);
                fzn_printer.add_output<int>(identifier, indices, callbacks);
            }
        }
    }

    // Arrays of bool variables
    for (auto const & entry : fzn_model.array_bool_vars)
    {
        auto const & identifier = entry.first;
        auto const & array_bool_var = entry.second;
        for (auto const & annotation : array_bool_var.annotations)
        {
            if (annotation.first == "output_array")
            {
                auto indices = get<vector<Fzn::int_range_t>>(annotation.second.at(0));
                auto callback = []() -> int {return rand() % 2  == 0;};
                vector<function<bool()>> callbacks(array_bool_var.variables.size(),  callback);
                fzn_printer.add_output<bool>(identifier, indices, callbacks);
            }
        }
    }
}