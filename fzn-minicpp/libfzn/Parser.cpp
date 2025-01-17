#include <fstream>

#include "Model.h"
#include "Parser.h"
#include "Utils.h"

Fzn::Parser::Parser()
{
    using namespace std;

    // Verbose parsing errors
    fzn_parser.set_logger([](size_t line, size_t col, string const & msg) {
        stringstream ss;
        ss << "Line " << line << " column " << col << " : " << msg;
        throw runtime_error(ss.str());
    });

    // Load FlatZinc grammar
    string const fzn_grammar =
        #include "fzn.peg"
    ;
    if (not fzn_parser.load_grammar(fzn_grammar.data(), fzn_grammar.size()))
    {
        throw runtime_error("Syntax error in fzn.peg");
    }

    // Add parsing actions
    add_constants_actions();
    add_literals_actions();
    add_identifiers_actions();
    add_ranges_sets_actions();
    add_types_actions();
    add_annotation_actions();
    add_expressions_actions();
    add_variables_actions();
    add_constraints_actions();
    add_solve_actions();
    add_search_annotations_actions();
}

Fzn::Model Fzn::Parser::parse(std::string const & fzn_file_path)
{
    using namespace std;

    Fzn::Model fzn_model;
    read_fzn_file(fzn_file_path, fzn_model.fzn_file_content);
    any data(&fzn_model);
    fzn_parser.parse_n(fzn_model.fzn_file_content.data(), fzn_model.fzn_file_content.size(), data);
    return fzn_model;
}

void Fzn::Parser::read_fzn_file(std::string const & fzn_file_path, std::vector<char> & buffer)
{
    using namespace std;

    if (fzn_file_path.compare(fzn_file_path.length() - 4, 4, ".fzn") != 0)
    {
        stringstream msg;
        msg << "Not a FlatZinc (.fzn) file : "<< fzn_file_path;
        throw runtime_error(msg.str());
    }

    ifstream ifs(fzn_file_path, ios::in | ios::binary);
    if (ifs.fail())
    {
        stringstream ss;
        ss << fzn_file_path << " : unable to open";
        throw runtime_error(ss.str());
    }

    auto const size = ifs.seekg(0, ios::end).tellg();
    buffer.resize(size);
    ifs.seekg(0, ios::beg).read(buffer.data(), size);
}


void Fzn::Parser::add_constants_actions()
{
    using namespace std;
    using namespace peg;

    // basic_constant <- basic_literal_type ":" identifier "=" basic_literal_expr ";"
    fzn_parser["basic_constant"] = [](SemanticValues const & vs)
    {
        stringstream ss;
        auto const & li = vs.line_info();
        ss << "Line " << li.first << " column " << li.second << " : " << "single constants are not supported";
        throw runtime_error(ss.str());
    };

    // array_constant <- array_literal_type ":" identifier "=" array_literal_expr ";"
    fzn_parser["array_constant"] = [](SemanticValues const & vs, any & dt)
    {
        auto * const fzn_model = any_cast<Fzn::Model*>(dt);
        auto const & identifier = any_cast<identifier_t>(vs.at(1));
        auto array_literal_expr = any_cast<array_literal_expr_t>(vs.at(2));
        if (std::holds_alternative<vector<int>>(array_literal_expr))
        {
            auto & array = std::get<vector<int>>(array_literal_expr);
            fzn_model->array_int_consts.emplace(std::move(string(identifier)), std::move(array));
            return;
        }
        if (std::holds_alternative<vector<bool>>(array_literal_expr))
        {
            auto & array = std::get<vector<bool>>(array_literal_expr);
            fzn_model->array_bool_consts.emplace(std::move(string(identifier)), std::move(array));
            return;
        }
        stringstream ss;
        auto const & li = vs.line_info();
        ss << "Line " << li.first << " column " << li.second << " : " << "unsupported array of constants";
        throw runtime_error(ss.str());

    };
}

void Fzn::Parser::add_variables_actions()
{
    using namespace std;
    using namespace peg;

    // basic_variable <- basic_var_type ":" identifier annotations";"
    fzn_parser["basic_variable"] = [](SemanticValues const & vs, any & dt)
    {
        auto * const fzn_model = any_cast<Fzn::Model*>(dt);
        auto basic_var_type = any_cast<basic_var_type_t>(vs.at(0));
        auto const & identifier = any_cast<identifier_t>(vs.at(1));
        auto annotations = any_cast<vector<annotation_t>>(vs.at(2));

        Var var;
        var.annotations = std::move(annotations);
        if (std::holds_alternative<int_range_t>(basic_var_type))
        {
            var.domain = std::get<int_range_t>(basic_var_type);
            fzn_model->int_vars.emplace(std::move(string(identifier)), std::move(var));
            return;
        }
        if (std::holds_alternative<int_set_t>(basic_var_type))
        {
            var.domain = std::get<int_set_t>(basic_var_type);
            fzn_model->int_vars.emplace(std::move(string(identifier)), std::move(var));
            return;
        }
        if (std::holds_alternative<string_view>(basic_var_type) and std::get<string_view>(basic_var_type) == "bool")
        {
            var.domain = bool_set_t{false,true};
            fzn_model->bool_vars.emplace(std::move(string(identifier)), std::move(var));
            return;
        }
        stringstream ss;
        auto const & li = vs.line_info();
        ss << "Line " << li.first << " column " << li.second << " : " << "unsupported basic variable";
        throw runtime_error(ss.str());
    };

    // array_variable <- array_var_type ":" identifier annotations "=" array_var_expr ";"
    fzn_parser["array_variable"] = [](SemanticValues const & vs, any & dt)
    {
        auto * const fzn_model = any_cast<Fzn::Model*>(dt);
        auto const & array_var_type = any_cast<array_var_type_t>(vs.at(0));
        auto const & basic_var_type = array_var_type.second;
        auto const & identifier = any_cast<identifier_t>(vs.at(1));
        auto annotations = any_cast<vector<annotation_t>>(vs.at(2));
        auto array_var_expr = any_cast<vector<basic_var_expr_t>>(vs.at(3));

        ArrayVar array;
        array.variables = std::move(array_var_expr);
        array.annotations = std::move(annotations);
        if (std::holds_alternative<string_view>(basic_var_type) and std::get<string_view>(basic_var_type) == "int")
        {
            fzn_model->array_int_vars.emplace(std::move(string(identifier)), std::move(array));
            return;
        }
        if (std::holds_alternative<string_view>(basic_var_type) and std::get<string_view>(basic_var_type) == "bool")
        {
            fzn_model->array_bool_vars.emplace(std::move(string(identifier)), std::move(array));
            return;
        }
        stringstream ss;
        auto const & li = vs.line_info();
        ss << "Line " << li.first << " column " << li.second << " : " << "unsupported array type";
        throw runtime_error(ss.str());
    };
}

void Fzn::Parser::add_constraints_actions()
{
    using namespace std;
    using namespace peg;

    // constraint_arg <- basic_literal_expr / array_literal_expr / identifier / array_var_expr
    fzn_parser["constraint_arg"] = [](SemanticValues const & vs, any & dt)
    {

        if (isType<int>(vs.at(0)))
        {
            constraint_arg_t arg {any_cast<int>(vs.at(0))};
            return arg;
        }
        if (isType<bool>(vs.at(0)))
        {
            constraint_arg_t arg {any_cast<bool>(vs.at(0))};
            return arg;
        }
        if (isType<array_literal_expr_t>(vs.at(0)))
        {
            auto array = any_cast<array_literal_expr_t>(vs.at(0));
            if (std::holds_alternative<std::monostate>(array))
            {
                return constraint_arg_t{};
            }
            if (std::holds_alternative<vector<int>>(array))
            {
                constraint_arg_t arg {std::move(get<vector<int>>(array))};
                return arg;
            }
            if (std::holds_alternative<vector<bool>>(array))
            {
                constraint_arg_t arg {std::move(get<vector<bool>>(array))};
                return arg;
            }
        }
        if (isType<identifier_t>(vs.at(0)))
        {
            constraint_arg_t arg {any_cast<identifier_t>(vs.at(0))};
            return arg;
        }
        if (isType<vector<basic_var_expr_t>>(vs.at(0)))
        {
            constraint_arg_t arg{any_cast<vector<basic_var_expr_t>>(vs.at(0))};
            return arg;
        }
        stringstream ss;
        auto const & li = vs.line_info();
        ss << "Line " << li.first << " column " << li.second << " : " << "unsupported constraint argument";
        throw runtime_error(ss.str());
    };

    //constraint_args <- "(" sequence(constraint_arg) ")"
    fzn_parser["constraint_args"] = [](SemanticValues const & vs)
    {
        return sequence_action<constraint_arg_t>(vs);
    };

    //constraint <- "constraint" pred_identifier constraint_args annotations ";"
    fzn_parser["constraint"] = [](SemanticValues const & vs, any & dt)
    {
        auto * const fzn_model = any_cast<Fzn::Model*>(dt);
        auto const & pred_identifier = any_cast<pred_identifier_t>(vs.at(0));
        auto const & constraint_args = any_cast<vector<constraint_arg_t>>(vs.at(1));
        auto const & annotations = any_cast<vector<annotation_t>>(vs.at(2));
        Constraint c;
        c.identifier = pred_identifier;
        c.arguments = constraint_args;
        c.annotations = annotations;
        fzn_model->constraints.emplace_back(std::move(c));
    };
}

void Fzn::Parser::add_literals_actions()
{
    using namespace std;
    using namespace peg;

    fzn_parser["int_literal"] = [](SemanticValues const & vs)
    {
        auto const & matched_string = vs.token();
        int int_literal;
        from_chars(matched_string.data(), matched_string.data() + matched_string.size(), int_literal);
        return int_literal;
    };

    fzn_parser["float_literal"] = [](SemanticValues const & vs)
    {
        auto const & matched_string = vs.token();
        float float_literal;
        from_chars(matched_string.data(), matched_string.data() + matched_string.size(), float_literal);
        return float_literal;
    };

    fzn_parser["bool_literal"] = [](SemanticValues const & vs)
    {
        auto const & bool_literal = vs.token();
        return bool_literal == "true";
    };
}

void Fzn::Parser::add_identifiers_actions()
{
    using namespace std;
    using namespace peg;

    fzn_parser["identifier"] = [](SemanticValues const & vs)
    {
        identifier_t const & identifier = vs.token();
        return identifier;
    };

    fzn_parser["pred_identifier"] = [](SemanticValues const & vs)
    {
        identifier_t const & pred_identifier = vs.token();
        return pred_identifier;
    };
}

void Fzn::Parser::add_ranges_sets_actions()
{
    using namespace std;
    using namespace peg;

    // int_range <- int_literal ".." int_literal
    fzn_parser["int_range"] = [](SemanticValues const & vs)
    {
        auto const min = any_cast<int>(vs.at(0));
        auto const max = any_cast<int>(vs.at(1));
        auto const int_range = int_range_t(min, max);
        return int_range;
    };

    // int_ranges <- sequence(int_range)
    fzn_parser["int_ranges"] = [](SemanticValues const & vs)
    {
        return sequence_action<int_range_t>(vs);
    };

    // int_set <- "{" sequence(int_literal) "}"
    fzn_parser["int_set"] = [](SemanticValues const & vs)
    {
        return sequence_action<int_set_t>(vs);
    };
}

void Fzn::Parser::add_types_actions()
{
    using namespace std;
    using namespace peg;

    //basic_literal_type <- "bool" / "int"
    fzn_parser["basic_literal_type"] = [](SemanticValues const & vs)
    {
        basic_literal_type_t const & basic_literal_type = vs.token();
        return basic_literal_type;
    };

    // array_literal_type <- "array" "[" int_range "]" "of" basic_literal_type
    fzn_parser["array_literal_type"] = [](SemanticValues const & vs)
    {
        auto int_range = any_cast<int_range_t>(vs.at(0));
        auto basic_literal_type = any_cast<basic_literal_type_t>(vs.at(1));
        array_literal_type_t array_literal_type(std::move(int_range), basic_literal_type);
        return array_literal_type;
    };

    //basic_var_type <- "var" (basic_literal_type / int_range / int_set)
    fzn_parser["basic_var_type"] = [](SemanticValues const & vs)
    {
        if (isType<basic_literal_type_t>(vs.at(0)))
        {
            basic_var_type_t type{any_cast<basic_literal_type_t>(vs.at(0))};
            return type;
        }
        if (isType<int_range_t>(vs.at(0)))
        {
            basic_var_type_t type{std::move(any_cast<int_range_t>(vs.at(0)))};
            return type;
        }
        if (isType<int_set_t>(vs.at(0)))
        {
            basic_var_type_t type{std::move(any_cast<int_set_t>(vs.at(0)))};
            return type;
        }
        stringstream ss;
        auto const & li = vs.line_info();
        ss << "Line " << li.first << " column " << li.second << " : " << "unsupported variable type";
        throw runtime_error(ss.str());
    };

    // array_var_type <- "array" "[" int_range "]" "of" basic_var_type
    fzn_parser["array_var_type"] = [](SemanticValues const & vs)
    {
        auto int_range = any_cast<int_range_t>(vs.at(0));
        auto basic_var_type = any_cast<basic_var_type_t>(vs.at(1));
        array_var_type_t array_var_type(std::move(int_range), std::move(basic_var_type));
        return array_var_type;
    };

    fzn_parser["solve_type"] = [](SemanticValues const & vs)
    {
        solve_type_t const & solve_type = vs.token();
        return solve_type;
    };
}

void Fzn::Parser::add_annotation_actions()
{
    using namespace std;
    using namespace peg;

    //annotation_arg <- int_literal / float_literal / identifier / "[" int_ranges "]"
    fzn_parser["annotation_arg"] = [](SemanticValues const & vs)
    {
        if (isType<int>(vs.at(0)))
        {
            annotation_arg_t arg{any_cast<int>(vs.at(0))};
            return arg;
        }
        if (isType<float>(vs.at(0)))
        {
            annotation_arg_t arg{any_cast<float>(vs.at(0))};
            return arg;
        }
        if (isType<identifier_t>(vs.at(0)))
        {
            annotation_arg_t arg{any_cast<identifier_t>(vs.at(0))};
            return arg;
        }
        if (isType<vector<int_range_t>>(vs.at(0)))
        {
            annotation_arg_t arg{std::move(any_cast<vector<int_range_t>>(vs.at(0)))};
            return arg;
        }
        stringstream ss;
        auto const & li = vs.line_info();
        ss << "Line " << li.first << " column " << li.second << " : " << "unsupported annotation argument";
        throw runtime_error(ss.str());
    };

    // annotation_args <- ("(" sequence(annotation_arg) ")") ?
    fzn_parser["annotation_args"] = [](SemanticValues const & vs)
    {
        return sequence_action<annotation_arg_t>(vs);
    };

    // annotation <- pred_identifier annotation_args
    fzn_parser["annotation"] = [](SemanticValues const & vs)
    {
        auto pred_identifier = any_cast<identifier_t>(vs.at(0));
        auto annotation_args = any_cast<vector<annotation_arg_t>>(vs.at(1));
        annotation_t annotation(pred_identifier, std::move(annotation_args));
        return annotation;
    };

    // annotations <- ("::" annotation)*
    fzn_parser["annotations"] = [](SemanticValues const & vs)
    {
        return sequence_action<annotation_t>(vs);
    };
}

void Fzn::Parser::add_expressions_actions()
{
    using namespace std;
    using namespace peg;

    // array_literal_expr <- "[" sequence(basic_literal_expr) "]"
    fzn_parser["array_literal_expr"] = [](SemanticValues const & vs)
    {
        if (vs.size() == 0)
        {
            return array_literal_expr_t{};
        }
        else if (isType<int>(vs.at(0)))
        {
            array_literal_expr_t array_literal_expr{std::move(sequence_action<int>(vs))};
            return array_literal_expr;
        }
        else if (isType<bool>(vs.at(0)))
        {
            array_literal_expr_t array_literal_expr{std::move(sequence_action<bool>(vs))};
            return array_literal_expr;
        }
        else
        {
            stringstream ss;
            auto const & li = vs.line_info();
            ss << "Line " << li.first << " column " << li.second << " : " << "unsupported array of literal expressions";
            throw runtime_error(ss.str());
        }

    };

    // fixed_var_expr <- basic_literal_expr
    fzn_parser["fixed_var_expr"] = [](SemanticValues const & vs)
    {
        if (isType<int>(vs.at(0)))
        {

            fixed_var_expr_t fixed_var_expr{vs.token(), any_cast<int>(vs.at(0))};
            return std::move(fixed_var_expr);
        }
        if (isType<bool>(vs.at(0)))
        {
            fixed_var_expr_t fixed_var_expr{vs.token(), any_cast<bool>(vs.at(0))};
            return std::move(fixed_var_expr);
        }
        stringstream ss;
        auto const & li = vs.line_info();
        ss << "Line " << li.first << " column " << li.second << " : " << "unsupported fixed variable expressions";
        throw runtime_error(ss.str());
    };

    // basic_var_expr <- identifier / fixed_var_expr
    fzn_parser["basic_var_expr"] = [](SemanticValues const & vs, any & dt)
    {
        auto * const fzn_model = any_cast<Fzn::Model*>(dt);
        if (isType<basic_var_expr_t>(vs.at(0)))
        {
            return any_cast<basic_var_expr_t>(vs.at(0));
        }
        if (isType<fixed_var_expr_t>(vs.at(0)))
        {
            auto const &fixed_var_expr = any_cast<fixed_var_expr_t>(vs.at(0));
            if (std::holds_alternative<bool>(fixed_var_expr.second))
            {
                auto const bool_val = std::get<bool>(fixed_var_expr.second);
                string bool_id(fixed_var_expr.first);
                if (fzn_model->bool_vars.count(bool_id) == 0)
                {
                    Var var;
                    var.domain = bool_set_t{bool_val};
                    fzn_model->bool_vars.emplace(std::move(bool_id), std::move(var));
                }
                return any_cast<basic_var_expr_t>(fixed_var_expr.first);
            }
            if (std::holds_alternative<int>(fixed_var_expr.second))
            {
                auto const int_val = std::get<int>(fixed_var_expr.second);
                string int_id(fixed_var_expr.first);
                if (fzn_model->int_vars.count(int_id) == 0)
                {
                    Var var;
                    var.domain = int_range_t{int_val, int_val};
                    fzn_model->int_vars.emplace(std::move(int_id), std::move(var));
                }
                return any_cast<basic_var_expr_t>(fixed_var_expr.first);
            }
        }
        stringstream ss;
        auto const & li = vs.line_info();
        ss << "Line " << li.first << " column " << li.second << " : " << "unsupported basic variable expressions";
        throw runtime_error(ss.str());
    };

    // array_var_expr <- "[" sequence(basic_var_expr) "]"
    fzn_parser["array_var_expr"] = [](SemanticValues const & vs)
    {
        return sequence_action<basic_var_expr_t>(vs);
    };

    // var_expr <- basic_var_expr / array_var_expr
    fzn_parser["var_expr"] = [](SemanticValues const & vs)
    {
        if (isType<basic_var_expr_t>(vs.at(0)))
        {
            var_expr_t expr{any_cast<basic_var_expr_t>(vs.at(0))};
            return expr;
        }
        if (isType<vector<basic_var_expr_t>>(vs.at(0)))
        {
            var_expr_t expr{std::move(any_cast<vector<basic_var_expr_t>>(vs.at(0)))};
            return expr;
        }
        stringstream ss;
        auto const & li = vs.line_info();
        ss << "Line " << li.first << " column " << li.second << " : " << "unsupported variable expression";
        throw runtime_error(ss.str());
    };
}

void Fzn::Parser::add_solve_actions()
{
    using namespace std;
    using namespace peg;

    // solve <- "solve" search_annotations solve_type identifier? ";"
    fzn_parser["solve"] = [&](SemanticValues const & vs, any & dt) {
        auto * const fzn_model = any_cast<Fzn::Model*>(dt);
        auto search_annotations = any_cast<vector<search_annotation_t>>(vs.at(0));
        auto const & solve_type = any_cast<solve_type_t>(vs.at(1));
        fzn_model->search_strategy = std::move(search_annotations);
        fzn_model->solve_type = solve_type;
        if (vs.size() == 3)
        {
            fzn_model->objective_var = any_cast<identifier_t>(vs.at(2));
        }
    };

    fzn_parser["solve_type"] = [&](SemanticValues const & vs)
    {
        solve_type_t const & solve_type = vs.token();
        return solve_type;
    };
}

void Fzn::Parser::add_search_annotations_actions()
{
    using namespace std;
    using namespace peg;

    // basic_search_annotation <- pred_identifier "(" var_expr sequence(annotation)")"
    fzn_parser["basic_search_annotation"] = [](SemanticValues const & vs)
    {
        auto const & pred_identifier = any_cast<identifier_t>(vs.at(0));
        auto const & var_expr = any_cast<var_expr_t>(vs.at(1));

        auto annotations_count = vs.size() - 2;
        vector<annotation_t> annotations(annotations_count);
        for (auto i = 0; i < annotations_count; i += 1)
        {
            annotations.at(i) = std::move(any_cast<annotation_t>(vs.at(2 + i)));
        }
        basic_search_annotation_t basic_search_annotation{pred_identifier, var_expr, std::move(annotations)};
        return basic_search_annotation;
    };

    // array_search_annotation <- pred_identifier "(" "[" sequence(basic_search_annotation) "]" ")"
    fzn_parser["array_search_annotation"] = [](SemanticValues const & vs)
    {
        auto const & pred_identifier = any_cast<identifier_t>(vs.at(0));
        auto const annotations_count = vs.size() - 1;
        vector<basic_search_annotation_t> basic_search_annotations(annotations_count);
        for (auto i = 0; i < annotations_count; i +=1)
        {
            basic_search_annotations.at(i) = std::move(any_cast<basic_search_annotation_t>(vs.at(1 + i)));
        }
        array_search_annotation_t array_search_annotation{pred_identifier,std::move(basic_search_annotations)};
        return array_search_annotation;
    };

    // search_annotation <- basic_search_annotation / array_search_annotation
    fzn_parser["search_annotation"] = [](SemanticValues const & vs)
    {
        if (isType<basic_search_annotation_t>(vs.at(0)))
        {
            auto basic_search_annotation = any_cast<basic_search_annotation_t>(vs.at(0));
            search_annotation_t search_annotation{std::move(basic_search_annotation)};
            return search_annotation;
        }
        if (isType<array_search_annotation_t>(vs.at(0)))
        {
            auto array_search_annotation = any_cast<array_search_annotation_t>(vs.at(0));
            search_annotation_t search_annotation{std::move(array_search_annotation)};
            return search_annotation;
        }
        stringstream ss;
        auto const & li = vs.line_info();
        ss << "Line " << li.first << " column " << li.second << " : " << "unsupported search annotation";
        throw runtime_error(ss.str());
    };

    // search_annotations <- ("::" search_annotation)*
    fzn_parser["search_annotations"] = [](SemanticValues const & vs)
    {
        return sequence_action<search_annotation_t>(vs);
    };
}