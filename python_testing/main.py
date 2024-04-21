from utils.io import parse_input
import utils.printer as printer

def main(input_file_path = "input.txt"):
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