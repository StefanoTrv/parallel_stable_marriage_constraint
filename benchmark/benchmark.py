import math
import subprocess
import random
import shutil
import os
import sys

def main():
    config = open("config.txt","r")
    empty_output_folder()
    #line format: <instance_size> <number_of_tests> <force_order_of_binding> <double_constraint> <rnd_tenth_domain>
    i = 0
    for l in config:
        l = l.split("\n")[0]
        n :int = int(l.split(" ")[0])
        number_of_tests : int = int(l.split(" ")[1])
        force_order : bool = l.split(" ")[2].upper() == "TRUE"
        double_constraint : bool = l.split(" ")[3].upper() == "TRUE"
        rnd_tenth_domain : bool = l.split(" ")[4].upper() == "TRUE"
        output_file_serial = open("out/serial_benchmark_tests_"+str(i)+".txt","w")
        output_file_parallel = open("out/parallel_benchmark_tests_"+str(i)+".txt","w")
        print("Beginning benchmark of "+str(number_of_tests)+" tests of size n="+str(n)+", with force_order="+str(force_order)+", double_constraint="+str(double_constraint)+", rnd_tenth_domain="+str(rnd_tenth_domain))
        run_tests(n,number_of_tests,force_order, double_constraint, rnd_tenth_domain, output_file_serial, output_file_parallel)
        print("Completed "+str(number_of_tests)+" tests.\n")
        output_file_serial.close()
        output_file_parallel.close()
        i = i + 1
    print("Completed all tests.")

def run_tests(n : int, number_of_tests : int, force_order : bool, double_constraint : bool, rnd_tenth_domain : bool, output_file_serial, output_file_parallel):
    for i in range(number_of_tests):
        write_input_files(n,force_order, double_constraint, rnd_tenth_domain)
        result_serial = subprocess.run("minizinc --solver ../fzn-minicpp/org.minicpp.minicpp.msc serial_input.mzn --output-time", shell=True, capture_output=True, text=True)
        serial_output = result_serial.stdout
        serial_error_string = result_serial.stderr
        #if serial_error_string != "" or "ERROR" in serial_output: #commented because of issues with minizinc's latest release
        if "ERROR" in serial_output:
            print("Execution of solver with serial constraint ended in error! Terminating...\nError details:\n"+serial_error_string)
            sys.exit(1)
        result_parallel = subprocess.run("minizinc --solver ../fzn-minicpp/org.minicpp.minicpp.msc parallel_input.mzn --output-time", shell=True, capture_output=True, text=True)
        parallel_output = result_parallel.stdout
        parallel_error_string = result_parallel.stderr
        #if parallel_error_string != "" or "ERROR" in parallel_output: #commented because of issues with minizinc's latest release
        if "ERROR" in parallel_output:
            print("Execution of solver with parallel constraint ended in error! Terminating...\nError details:\n"+parallel_error_string)
            sys.exit(1)
        if serial_output.split("time elapsed")[0]!=parallel_output.split("time elapsed")[0]:
            print("Found error!")
            print(serial_output)
            print(parallel_output)
            exit()
        output_file_serial.write(serial_output.split("time elapsed")[1].split("----------")[0])
        output_file_parallel.write(parallel_output.split("time elapsed")[1].split("----------")[0])
        

def write_input_files(n : int, force_order : bool, double_constraint : bool, rnd_tenth_domain : bool):
    serial_input = open("serial_input.mzn", "w")
    parallel_input = open("parallel_input.mzn", "w")
    serial_input.write("include \"stable_matching.mzn\";\ninclude \"minicpp.mzn\";\n\n")
    parallel_input.write("include \"stable_matching.mzn\";\ninclude \"minicpp.mzn\";\n\n")

    #n
    serial_input.write("int: n = "+str(n)+";\n\n")
    parallel_input.write("int: n = "+str(n)+";\n\n")

    #men's variables
    for i in range(n):
        serial_input.write("var 0..n-1 : m"+str(i)+";\n")
        parallel_input.write("var 0..n-1 : m"+str(i)+";\n")

    #women's variables
    for i in range(n):
        serial_input.write("var 0..n-1 : w"+str(i)+";\n")
        parallel_input.write("var 0..n-1 : w"+str(i)+";\n")

    #men's preference list
    serial_input.write("array [int,int] of 0..n-1: pm =\n[|")
    parallel_input.write("array [int,int] of 0..n-1: pm =\n[|")
    preference_table = random_preference_table(n)
    serial_input.write("|\n  ".join(map(lambda pt: ",".join(map(str, pt)), preference_table)))
    parallel_input.write("|\n  ".join(map(lambda pt: ",".join(map(str, pt)), preference_table)))
    serial_input.write("|];\n\n")
    parallel_input.write("|];\n\n")

    #women's preference list
    serial_input.write("array [int,int] of 0..n-1: pw =\n[|")
    parallel_input.write("array [int,int] of 0..n-1: pw =\n[|")
    preference_table = random_preference_table(n)
    serial_input.write("|\n  ".join(map(lambda pt: ",".join(map(str, pt)), preference_table)))
    parallel_input.write("|\n  ".join(map(lambda pt: ",".join(map(str, pt)), preference_table)))
    serial_input.write("|];\n\n")
    parallel_input.write("|];\n\n")
    
    men_var_list_string = ",".join(f"m{i}" for i in range(n))
    women_var_list_string = ",".join(f"w{i}" for i in range(n))

    serial_input.write("constraint stable_matching(["+men_var_list_string+"], ["+women_var_list_string+"], pm, pw) ::uniud;\n\n")
    parallel_input.write("constraint stable_matching(["+men_var_list_string+"], ["+women_var_list_string+"], pm, pw) ::gpu;\n\n")

    if double_constraint:
        serial_input.write("constraint stable_matching(["+women_var_list_string+"], ["+men_var_list_string+"], pw, pm) ::uniud;\n\n")
        parallel_input.write("constraint stable_matching(["+women_var_list_string+"], ["+men_var_list_string+"], pw, pm) ::gpu;\n\n")

    if rnd_tenth_domain:
        i : int = random.randint(0,n-1)
        p : str = random.choice(["m","w"])
        serial_input.write("constraint "+ p + str(i) +" > " + str(math.ceil(n*0.1)) + ";\n\n")
        parallel_input.write("constraint "+ p + str(i) +" > " + str(math.ceil(n*0.1)) + ";\n\n")

    if force_order:
            serial_input.write("solve :: seq_search([\n    int_search(["+men_var_list_string+"], input_order, indomain_min, complete),\n    int_search(["+women_var_list_string+"], input_order, indomain_max, complete)\n]) satisfy;\n")
            parallel_input.write("solve :: seq_search([\n    int_search(["+men_var_list_string+"], input_order, indomain_min, complete),\n    int_search(["+women_var_list_string+"], input_order, indomain_max, complete)\n]) satisfy;\n")
    else:
        serial_input.write("solve satisfy;\n")
        parallel_input.write("solve satisfy;\n")

    serial_input.close()
    parallel_input.close()

def random_preference_table(n : int) -> list[list[int]]:
    preference_table = []
    for i in range(n):
        preference_list = list(range(n))
        random.shuffle(preference_list)
        preference_table.append(preference_list)
    return preference_table

def empty_output_folder():
    shutil.rmtree("out", ignore_errors=True)
    os.makedirs("out")

if __name__ == "__main__":
    main()