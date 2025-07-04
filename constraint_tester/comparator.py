import math
import subprocess
import argparse
import random
import shutil
import os

def main():
    parser = argparse.ArgumentParser("comparator")
    parser.add_argument("n", help="The size of the test instances, an integer.", type=int)
    parser.add_argument("number_of_tests", help="The number of tests to be executed on full domains, an integer.", type=int)
    parser.add_argument("double_constraint", help="If there should be two mirrored constraints.", type=str)
    parser.add_argument("force_binding", help="If the binding order must be forced.", type=str)
    parser.add_argument("rnd_tenth_domain", help="If the first tenth of a random domain must be removed.", type=str)
    args = parser.parse_args()
    n = args.n
    number_of_tests = args.number_of_tests
    double_constraint = args.double_constraint.upper() == "TRUE"
    force_binding = args.force_binding.upper() == "TRUE"
    rnd_tenth_domain = args.rnd_tenth_domain.upper() == "TRUE"

    empty_output_folder()

    errors = 0
    for i in range(number_of_tests):
        write_input_files(n,True,double_constraint,force_binding,rnd_tenth_domain)
        result_serial = subprocess.run("minizinc --solver ../fzn-minicpp/org.minicpp.minicpp.msc serial_input.mzn", shell=True, capture_output=True, text=True)
        serial_output = result_serial.stdout
        serial_error_string = result_serial.stderr
        #if serial_error_string != "" or "ERROR" in serial_output: #commented because of issues with minizinc's latest release
        if "ERROR" in serial_output:
            print("Execution of solver with serial constraint ended in error! Terminating...\nError details:\n"+serial_error_string+"\nMinizinc output:\n"+serial_output)
            return
        result_parallel = subprocess.run("minizinc --solver ../fzn-minicpp/org.minicpp.minicpp.msc parallel_input.mzn", shell=True, capture_output=True, text=True)
        parallel_output = result_parallel.stdout
        parallel_error_string = result_parallel.stderr
        #if parallel_error_string != "" or "ERROR" in parallel_output: #commented because of issues with minizinc's latest release
        if "ERROR" in parallel_output:
            print("Execution of solver with parallel constraint ended in error! Terminating...\nError details:\n"+parallel_error_string+"\nMinizinc output:\n"+serial_output)
            return
        if(serial_output != parallel_output):
            shutil.copyfile("serial_input.mzn", "out/error_"+str(errors)+"_serial.mzn")
            shutil.copyfile("parallel_input.mzn", "out/error_"+str(errors)+"_parallel.mzn")
            error_file = open("out/error_"+str(errors)+"_outputs.txt","w")
            error_file.write("Serial output:\n"+serial_output+"\n\nParallel output:\n"+parallel_output)
            error_file.close()
            errors += 1
        if errors >= 20:
            print("Found 20 errors! Interrupting after "+str(i+1)+" tests.")
            return
        if i%25==0:
            print("Completed test "+str(i)+" out of "+str(number_of_tests)+".")
    if errors == 0:
        print("Completed "+str(number_of_tests)+" tests without finding any error!")
    else:
        print("Testing completed. "+str(errors)+" errors were found.")
        

def write_input_files(n : int, full_domains : bool, double_constraint : bool, force_binding : bool, rnd_tenth_domain : bool):
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

    if force_binding:
        serial_input.write("solve :: seq_search([\n    int_search(["+women_var_list_string+"], input_order, indomain_max, complete),\n    int_search(["+men_var_list_string+"], input_order, indomain_min, complete)\n]) satisfy;\n")
        parallel_input.write("solve :: seq_search([\n    int_search(["+women_var_list_string+"], input_order, indomain_max, complete),\n    int_search(["+men_var_list_string+"], input_order, indomain_min, complete)\n]) satisfy;\n")
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