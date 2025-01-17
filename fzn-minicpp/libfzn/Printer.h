#pragma once

#include <any>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <ostream>

namespace Fzn
{
    class Output
    {
        public:
            using int_range_t = std::pair<int, int>;

        public:
            std::string const identifier;
            std::vector<int_range_t> const indices;
            std::vector<std::any> get_value_callbacks;

        public:
            template<typename T>
            Output(std::string & identifier, std::function<T()> & get_value);
            template<typename T>
            Output(std::string & identifier,
                   std::vector<int_range_t> & indices,
                   std::vector<std::function<T()>> & get_values);
    };

    class Printer
    {
        public:
            using int_range_t = std::pair<int, int>;

        private:
            std::vector<Output> outputs;

        public:
            Printer();
            template<typename T>
            void add_output(std::string identifier, std::function<T()> get_value);
            template<typename T>
            void add_output(std::string identifier,
                            std::vector<int_range_t> & indices,
                            std::vector<std::function<T()>> & get_values);
            void print_outputs(std::ostream & os) const;
        private:
            static void print_value(std::any const & callback, std::ostream & os);
    };
}

template<typename T>
Fzn::Output::Output(std::string & identifier, std::function<T()> & get_value) :
        identifier(std::move(identifier)),
        indices(),
        get_value_callbacks()
{
    get_value_callbacks.emplace_back(std::move(get_value));
}

template<typename T>
Fzn::Output::Output(std::string & identifier,
               std::vector<int_range_t> & indices,
               std::vector<std::function<T()>> & get_values) :
        identifier(std::move(identifier)),
        indices(std::move(indices)),
        get_value_callbacks()
{
    int values = get_values.size();
    for (int i = 0; i < values; i += 1)
    {
        get_value_callbacks.emplace_back(std::move(get_values.at(i)));
    }
}

template<typename T>
void Fzn::Printer::add_output(std::string identifier, std::function<T()> get_value)
{
    outputs.emplace_back(identifier, get_value);
}

template<typename T>
void Fzn::Printer::add_output(std::string identifier,
                         std::vector<int_range_t> & indices,
                         std::vector<std::function<T()>> & get_values)
{
    outputs.emplace_back(identifier, indices, get_values);
}
