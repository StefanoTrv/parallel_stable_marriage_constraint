from utils.io import parse_input
import utils.printer as printer
import constraint_functions as cs
from domain_functions import *
import sys
 

def main():
    #read command line arguments
    input_file_path = "input.txt"
    if len(sys.argv)>1:
        input_file_path = sys.argv[1]
    print(input_file_path)

    # parse the input from the txt file
    n, xpl, ypl, x_domain, y_domain = parse_input(input_file_path)

    #initializes the full domains if they're not initialized
    if len(x_domain)==0:
        for i in range(n):
            x_domain.append([])
            y_domain.append([])
            for j in range(n):
                x_domain[i].append(True)
                y_domain[i].append(True)
    
    printer.print_preference_lists(n,xpl,ypl)

    printer.print_domains(n,x_domain,y_domain)

    #computes the reverse matrixes
    xPy = get_reverse_matrix(n,xpl)
    yPx = get_reverse_matrix(n,ypl)

    printer.print_reverse_matrixes(n,xPy,yPx)

    #initializes xlb and yub
    xlb = [0] * n
    yub = [n-1] * n

    #applies once the constraint
    old_x_domain = [l.copy() for l in x_domain]
    old_y_domain = [l.copy() for l in y_domain]
    prev_x_domain = [l.copy() for l in x_domain]
    prev_y_domain = [l.copy() for l in y_domain]


    cs.init(n,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb)
    while True:
        stop = True
        for i in range(n):
            if getMin(n,x_domain,i)!=getMin(n,old_x_domain,i):
                cs.deltaMin(i,n,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb)
                print("deltaMin "+str(i))
                printer.print_domains(n,x_domain,y_domain)
                stop = False
            if getMax(n,y_domain,i)!=getMax(n,old_y_domain,i):
                cs.deltaMax(i,n,x_domain,y_domain,xpl,ypl,xPy,yPx,yub)
                print("deltaMax "+str(i))
                printer.print_domains(n,x_domain,y_domain)
                stop = False
            for k in range(getMin(n,x_domain,i)+1,getMax(n,x_domain,i)):
                if x_domain[i][k]!=old_x_domain[i][k]:
                    cs.removeValue(i,k,n,x_domain,y_domain,xpl,ypl,xPy,yPx)
                    print("removeValue "+str(i)+" "+str(k))
                    printer.print_domains(n,x_domain,y_domain)
                    stop = False
        
        if stop:
            break
        
        print("I have not stopped!")

        #updates old domains
        old_x_domain = prev_x_domain
        old_y_domain = prev_y_domain
        prev_x_domain = [l.copy() for l in x_domain]
        prev_y_domain = [l.copy() for l in y_domain]
    
    printer.print_domains(n,x_domain,y_domain)

    print("Men best: ")
    for i in range(n):
        print(xpl[i][getMin(n,x_domain,i)])

    return

def get_reverse_matrix(n,zpl):
    zPz = []
    for i in range(n):
        l = []
        for j in range(n):
            l.append(-1)
        for j in range(n):
            l[zpl[i][j]]=j
        zPz.append(l)
    return zPz

if __name__ == "__main__":
    main()