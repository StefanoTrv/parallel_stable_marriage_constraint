def parse_input(file_path):
    with open(file_path) as f:

        # get file size
        currentPos=f.tell()
        f.seek(0, 2)          # move to end of file
        file_size = f.tell()  # get current position
        f.seek(currentPos, 0) # go back to where we started

        #reads n and initializes arrays
        n = int(f.readline().split()[0])
        xpl = []
        ypl = []
        x_domain = []
        y_domain = []

        f.readline() # skips empty line

        for i in range(n): # fills xpl
            line=f.readline()
            xpl.append([int(x) for x in line.split()])
        
        f.readline() # skips empty line

        for i in range(n): # fills ypl
            line=f.readline()
            ypl.append([int(x) for x in line.split()])
        
        f.readline() # skips empty line

        if (file_size - f.tell() >= n*n*2): #fills domains only if present in the file
            for i in range(n): # read n lines
                line=f.readline()
                x_domain.append([bool(int(x)) for x in line.split()])
            
            f.readline() # skips empty line

            for i in range(n): # read n lines
                line=f.readline()
                y_domain.append([bool(int(x)) for x in line.split()])

        f.close()
        return n, xpl, ypl, x_domain, y_domain