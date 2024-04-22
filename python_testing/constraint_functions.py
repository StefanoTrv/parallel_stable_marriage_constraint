from domain_functions import * 

def inst(i,n,x_domain,y_domain,xpl,ypl,xPy,yPx):
    for k in range(getVal(n,x_domain,i)):
        j = xpl[i][k]
        setMax(n,y_domain,j,yPx[j][i]-1)
    
    j = xpl[i][getVal(n,x_domain,i)]
    setVal(n,y_domain,j,yPx[j][i])

    for k in range(getVal(n,x_domain,i)+1,n):
        j = xpl[i][k]
        remVal(y_domain,j,yPx[j][i])

def removeValue(i,a,n,x_domain,y_domain,xpl,ypl,xPy,yPx):
    j = xpl[i][a]
    remVal(y_domain,j,yPx[j][i])

def deltaMin(i,n,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb):
    j = xpl[i][getMin(n,x_domain,i)]
    setMax(n,y_domain,j,yPx[j][i])
    for k in range(xlb[i],getMin(n,x_domain,i)):
        j = xpl[i][k]
        setMax(n,y_domain,j,yPx[j][i]-1)
    xlb[i] = getMin(n,x_domain,i)

def deltaMax(j,n,x_domain,y_domain,xpl,ypl,xPy,yPx,yub):
    for k in range(getMax(n,y_domain,j)+1,yub[j]+1):
        i = ypl[j][k]
        remVal(n,x_domain,i,xPy[i][j])
    yub[j] = getMax(n,y_domain,j)

def init(n,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb):
    for i in range(n):
        deltaMin(i,n,x_domain,y_domain,xpl,ypl,xPy,yPx,xlb)