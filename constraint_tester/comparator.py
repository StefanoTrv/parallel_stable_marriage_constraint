import subprocess
import argparse
import random
import shutil
import os

def main():
    parser = argparse.ArgumentParser("comparator")
    parser.add_argument("n", help="The size of the test instances, an integer.", type=int)
    parser.add_argument("full_tests", help="The number of tests to be executed on full domains, an integer.", type=int)
    args = parser.parse_args()
    n = args.n
    full_tests = args.full_tests

    empty_output_folder()

    errors = 0
    for i in range(full_tests):
        write_input_files(n,True)
        result_serial = subprocess.run(["./fzn-minicpp", "serial_input.fzn"], capture_output=True, text=True)
        serial_output = result_serial.stdout
        serial_error_string = result_serial.stderr
        if serial_error_string != "":
            print("Execution of solver with serial constraint ended in error! Terminating...\nError details:\n"+serial_error_string)
            return
        result_parallel = subprocess.run(["./fzn-minicpp", "parallel_input.fzn"], capture_output=True, text=True)
        parallel_output = result_parallel.stdout
        parallel_error_string = result_parallel.stderr
        if parallel_error_string != "":
            print("Execution of solver with parallel constraint ended in error! Terminating...\nError details:\n"+parallel_error_string)
            return
        if(serial_output != parallel_output):
            shutil.copyfile("serial_input.fzn", "out/error_"+str(errors)+"_serial.fzn")
            shutil.copyfile("parallel_input.fzn", "out/error_"+str(errors)+"_parallel.fzn")
            error_file = open("out/error_"+str(errors)+"_outputs.txt","w")
            error_file.write("Serial output:\n"+serial_output+"\n\nParallel output:\n"+parallel_output)
            errors += 1
        if errors >= 20:
            print("Found 20 errors! Interrupting after "+str(i+1)+" tests.")
            return
        if i%25==0:
            print("Completed test "+str(i)+" out of "+str(full_tests)+".")
    if errors == 0:
        print("Completed "+str(full_tests)+" tests without finding any error!")
    else:
        print("Testing completed. "+str(errors)+" errors were found.")
        

def write_input_files(n : int, full_domains : bool):
    serial_input = open("serial_input.fzn", "w")
    parallel_input = open("parallel_input.fzn", "w")
    serial_input.write("predicate minicpp_stable_matching(array [int] of var int: m,array [int] of var int: w,array [int] of int: pm,array [int] of int: pw);\n")
    parallel_input.write("predicate minicpp_stable_matching(array [int] of var int: m,array [int] of var int: w,array [int] of int: pm,array [int] of int: pw);\n")

    #women's preference list
    serial_input.write("array [1.."+str(n*n)+"] of int: X_INTRODUCED_18_ = [")
    parallel_input.write("array [1.."+str(n*n)+"] of int: X_INTRODUCED_18_ = [")
    preference_table = random_preference_table(n)
    serial_input.write(",".join(map(str, preference_table)))
    parallel_input.write(",".join(map(str, preference_table)))
    serial_input.write("];\n")
    parallel_input.write("];\n")

    #men's preference list
    serial_input.write("array [1.."+str(n*n)+"] of int: X_INTRODUCED_19_ = [")
    parallel_input.write("array [1.."+str(n*n)+"] of int: X_INTRODUCED_19_ = [")
    preference_table = random_preference_table(n)
    serial_input.write(",".join(map(str, preference_table)))
    parallel_input.write(",".join(map(str, preference_table)))
    serial_input.write("];\n")
    parallel_input.write("];\n")

    for i in range(n):
        serial_input.write("var 0.."+str(n-1)+": m"+str(i)+":: output_var;\n")
        parallel_input.write("var 0.."+str(n-1)+": m"+str(i)+":: output_var;\n")

    for i in range(n):
        serial_input.write("var 0.."+str(n-1)+": w"+str(i)+":: output_var;\n")
        parallel_input.write("var 0.."+str(n-1)+": w"+str(i)+":: output_var;\n")
    
    women_var_list_string = ",".join(f"w{i}" for i in range(n))
    serial_input.write("array [1.."+str(n)+"] of var int: X_INTRODUCED_20_ ::var_is_introduced  = ["+women_var_list_string+"];\n")
    parallel_input.write("array [1.."+str(n)+"] of var int: X_INTRODUCED_20_ ::var_is_introduced  = ["+women_var_list_string+"];\n")
    
    men_var_list_string = ",".join(f"m{i}" for i in range(n))
    serial_input.write("array [1.."+str(n)+"] of var int: X_INTRODUCED_21_ ::var_is_introduced  = ["+men_var_list_string+"];\n")
    parallel_input.write("array [1.."+str(n)+"] of var int: X_INTRODUCED_21_ ::var_is_introduced  = ["+men_var_list_string+"];\n")

    serial_input.write("constraint minicpp_stable_matching(X_INTRODUCED_21_,X_INTRODUCED_20_,X_INTRODUCED_19_,X_INTRODUCED_18_):: uniud;\n")
    parallel_input.write("constraint minicpp_stable_matching(X_INTRODUCED_21_,X_INTRODUCED_20_,X_INTRODUCED_19_,X_INTRODUCED_18_):: gpu;\n")

    #serial_input.write("solve :: seq_search([\n    int_search(["+men_var_list_string+"], input_order, indomain_min, complete),\n    int_search(["+women_var_list_string+"], input_order, indomain_max, complete)\n]) satisfy;\n")
    #parallel_input.write("solve :: seq_search([\n    int_search(["+men_var_list_string+"], input_order, indomain_min, complete),\n    int_search(["+women_var_list_string+"], input_order, indomain_max, complete)\n]) satisfy;\n")

    serial_input.write("solve  satisfy;\n")
    parallel_input.write("solve  satisfy;\n")

    serial_input.close()
    parallel_input.close()

def random_preference_table(n : int):
    preference_table = []
    for i in range(n):
        preference_list = list(range(n))
        random.shuffle(preference_list)
        preference_table+=(preference_list)
    return preference_table

def empty_output_folder():
    shutil.rmtree("out", ignore_errors=True)
    os.makedirs("out")

if __name__ == "__main__":
    main()