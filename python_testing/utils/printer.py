def __print_inline(str):
    print(str,end="")

def print_preference_lists(n,xpl,ypl):
    #men
    __print_inline("Men preference lists:\n\n")
    __print_inline("_\t")
    for i in range(n):
        __print_inline(str(i)+"\t")
    for i in range(n):
        __print_inline("\n"+str(i)+":\t")
        for j in range(n):
            __print_inline(str(xpl[i][j])+"\t")
    
    __print_inline("\n")

    #women
    __print_inline("Women preference lists:\n\n")
    __print_inline("_\t")
    for i in range(n):
        __print_inline(str(i)+"\t")
    for i in range(n):
        __print_inline("\n"+str(i)+":\t")
        for j in range(n):
            __print_inline(str(ypl[i][j])+"\t")
            
    __print_inline("\n")

def print_domains(n,x_dom,y_dom):
    #men
    __print_inline("Men domains:\n\n")
    __print_inline("_\t")
    for i in range(n):
        __print_inline(str(i)+"\t")
    for i in range(n):
        __print_inline("\n"+str(i)+":\t")
        for j in range(n):
            __print_inline(str(int(x_dom[i][j]))+"\t")
    
    __print_inline("\n")

    #women
    __print_inline("Women domains:\n\n")
    __print_inline("_\t")
    for i in range(n):
        __print_inline(str(i)+"\t")
    for i in range(n):
        __print_inline("\n"+str(i)+":\t")
        for j in range(n):
            __print_inline(str(int(y_dom[i][j]))+"\t")
            
    __print_inline("\n")

def print_reverse_matrixes(n,xPy,yPx):
    #men
    __print_inline("Men reversed matrix:\n\n")
    __print_inline("_\t")
    for i in range(n):
        __print_inline(str(i)+"\t")
    for i in range(n):
        __print_inline("\n"+str(i)+":\t")
        for j in range(n):
            __print_inline(str(xPy[i][j])+"\t")
    
    __print_inline("\n")

    #women
    __print_inline("Women reversed matrix:\n\n")
    __print_inline("_\t")
    for i in range(n):
        __print_inline(str(i)+"\t")
    for i in range(n):
        __print_inline("\n"+str(i)+":\t")
        for j in range(n):
            __print_inline(str(yPx[i][j])+"\t")
            
    __print_inline("\n")