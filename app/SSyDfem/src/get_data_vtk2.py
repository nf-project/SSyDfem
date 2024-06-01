# Extracts input data from file
from pylab import *
import numpy as np

def get_data_vtk(filename,typ):
    """
    :param file_name: Data file name
    :return: Variable dictionary
    """

    f = open(filename, 'r')
    datadic = {}     # return data dictionary
    vin = False      # value extraction mode
    dtype = ''       # float or ndarray
    vname = ''       # variable name
    varray = []      # list array
    len_ = -1
    count = 0
    
    for line in f:
        ec = '\n,;[]'                                             # remove extra characters
        for c in ec:
            line = line.replace(c, ' ')
                
        line = line.strip()
        if line == '':                                            # ignore blank lines
            line = "INCOHERENCE XX"
        
        if not vin:
            line = line.split(" ")
            if len(line)>=1:
                if line[0]=='POINTS':
                    vin = True
                    dtype = 'ndarray'
                    vname = line[0]
                    len_ = int(line[1])
                    count=0
                    varray=[]
                elif line[0]=='CELLS':
                    vin = True
                    dtype = 'cells'
                    vname = line[0]
                    len_ = int(line[1])
                    count=0
                    varray=[]
                elif line[0]=='CELL_DATA':
                    vin = False
                    len_ = int(line[1])
                elif line[0]=='LOOKUP_TABLE':
                    vin = True
                    dtype = 'ndarray'
                    vname = line[0]
                    count=0
                    varray=[]
        else:
            line = line.split(" ")
            if dtype=='ndarray':
                if count < len_:
                    if len(line)==1:
                        varray.append(float(line[0]))
                    else:
                        varray.append([float(val) for val in line])
                    count +=1
                if count == len_:
                    datadic[vname ] = np.array(varray)
                    vin = False

            elif dtype=="cells":
                if count < len_:
                    varray.append([int(val) for val in line])
                count +=1
                if count == len_:
                    datadic[vname] = (varray)
                    elements_=[]
                    for i in range(len_):
                        if varray[i][0] == typ:
                            val = varray[i] [1:]
                            # val.sort()
                            elements_.append(val)
                    elements=np.array( elements_)
                    datadic["ELEMENTS"] = elements
                    varray = []
                    vin = False
    
    return datadic