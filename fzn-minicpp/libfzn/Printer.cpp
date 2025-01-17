#include "Utils.h"
#include "Printer.h"

Fzn::Printer::Printer()
{}

void Fzn::Printer::print_outputs(std::ostream & os) const
{
    using namespace std;

    for (auto const & output: outputs)
    {
        os << output.identifier << " = ";

        auto const values_count = output.get_value_callbacks.size();
        if (values_count == 1)
        {
            // Value
            auto const & callback = output.get_value_callbacks.at(0);
            print_value(callback, os);
            os << ";" << endl;
        }
        else
        {
            // Indices
            auto const indices_count = output.indices.size();
            os << "array" << indices_count << "d(";
            // First indices
            os << output.indices.at(0).first << ".." << output.indices.at(0).second;
            // Remaining indices
            for (auto i = 1; i < indices_count; i += 1)
            {
                os << "," << output.indices.at(i).first << ".." << output.indices.at(i).second;
            }

            // Values
            os << ",[";
            // First value
            auto const & callback = output.get_value_callbacks.at(0);
            print_value(callback, os);
            // Remaining values
            for (auto i = 1; i < values_count; i += 1)
            {
                os << ",";
                auto & callback = output.get_value_callbacks.at(i);
                print_value(callback, os);
            }
            os << "]);" << endl;
        }
    }
}

void Fzn::Printer::print_value(std::any const & callback, std::ostream & os)
{
    using namespace std;

    if (isType<function<bool()>>(callback))
    {
        bool const value = any_cast<function<bool()>>(callback)();
        os << (value ? "true" : "false");
    }
    else if (isType<function<int()>>(callback))
    {
        int const value = any_cast<function<int()>>(callback)();
        os << value;
    }
    else
    {
        throw runtime_error("Unsupported callback type");
    }
}


