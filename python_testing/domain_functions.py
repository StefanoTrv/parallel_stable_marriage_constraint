def getMin(n,domain,index):
    for i in range(n):
        if domain[index][i]:
            return i
    raise ValueError("Domain for variable " + str(index) + " is empty.")

def getMax(n,domain,index):
    for i in reversed(range(n)):
        if domain[index][i]:
            return i
    raise ValueError("Domain for variable " + str(index) + " is empty.")

def getVal(n,domain,index):
    m = getMax(n,domain,index)
    if m == getMin(n,domain,index):
        return m
    else:
        raise ValueError("Variable " + str(index) + " has not been instantiated.")

def setMax(n,domain,index,a):
    old_max = getMax(n,domain,index)
    if a < old_max:
        for i in range(a+1,old_max+1):
            domain[index][i] = False

def setVal(n,domain,index,a):
    for i in range(n):
        if i!=a:
            domain[index][i] = False
        else:
            domain[index][i] = True

def remVal(domain,index,a):
    domain[index][a] = False