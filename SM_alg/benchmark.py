import subprocess
import shutil
import os
import sys

def main():
    config = open("config.txt","r")
    empty_output_folder()
    #line format: <instance_size> <number_of_tests>
    i = 0
    for l in config:
        l = l.strip()
        n :int = int(l.split(" ")[0])
        number_of_tests : int = int(l.split(" ")[1])
        output_file = open("out/benchmark_tests_"+str(i)+".txt","w")
        print("Beginning benchmark of "+str(number_of_tests)+" tests of size n="+str(n))
        result = subprocess.run("./cmp_algs.out "+str(n)+" "+str(number_of_tests), shell=True, capture_output=True, text=True)
        output = result.stdout
        error_string = result.stderr
        if error_string != "" or "Found error!" in output:
            print("Execution of benchmarks ended in error! Terminating...\nError details:\n"+error_string)
            print(output)
            sys.exit(1)
        output_file.write(output)
        print("Completed "+str(number_of_tests)+" tests.\n")
        output_file.close()
        i = i + 1
    print("Completed all tests.")


def empty_output_folder():
    shutil.rmtree("out", ignore_errors=True)
    os.makedirs("out")

if __name__ == "__main__":
    main()