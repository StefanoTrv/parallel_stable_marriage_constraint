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
    
    #finds the current maxs and mins for all the variables
    min_x = []
    max_x = []
    min_y = []
    max_y = []
    for i in range(n):
        for j in range(n):
            if x_domain[i][j]:
                min_x.append(j)
                break
        for j in range(n):
            if y_domain[i][j]:
                min_y.append(j)
                break
        for j in reversed(range(n)):
            if x_domain[i][j]:
                max_x.append(j)
                break
        for j in reversed(range(n)):
            if y_domain[i][j]:
                max_y.append(j)
                break
    
    printer.print_preference_lists(n,xpl,ypl)

    printer.print_domains(n,x_domain,y_domain)

    return

if __name__ == "__main__":
    main()