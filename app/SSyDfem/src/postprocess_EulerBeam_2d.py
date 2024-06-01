from pylab import *
# from report_generation_plate_mzc import *
from scipy.interpolate import griddata as gd
import numpy as np
from pyvtk import *

def postprocess_EulerBeam_2d(prop, dimen ,coordinates, elements, u, reaction, Strnod, file_name):
    """
    :param young : Young's modulus
    :param poiss : Poisson's ration
    :param thick : Plate thickness
    :param denss : Plate material density
    :param coordinates:  Coordinate matrix nnode x ndime (2)
    :param u: 5 per Nodal displacements
    :param reaction: Force and Moment Nodal reactions
    :param Strnod: Local Moment, Locak Membrane and Nodal stresses
    :param la: X, Y and Z Vectors OnGaussPoints GPTR
    :param file_name: Save file name
    :return: Void
    """

    file_name = "output/"+file_name

    # Number of nodes
    [npnod, nelem, gl] = dimen

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    # z = coordinates[:, 2]
        
    u_xyz = []
    du_xyz = []
    
    theta_xyz = []
    
    for i in range(npnod):
                
        du_xyz.append([u[i * gl, 0],u[i * gl+1, 0],0.0 ])
        u_xyz.append([x[i]+u[i * gl, 0],y[i]+u[i * gl+1, 0], 0.0 ])
        theta_xyz.append([u[i * gl + 2, 0],0.0,0.0])
    
    """
    Reactions
    """
    f_xyz = []
    m_xyz = []
    for i in range(npnod):
        f_xyz.append( [ reaction[i * gl, 0],reaction[i * gl+1, 0], 0.0 ] )
        
        m_xyz.append( [ reaction[i * gl+2, 0], 0.0 , 0.0 ] )
    
    """
    Stress Axial and Bending
    """
    
    sa  = Strnod[:, 0]
    sb1  = Strnod[:, 1]
    sb2 = Strnod[:, 2]
    s1=[]
    
    """
    Stress max and min
    """
    smin  = Strnod[:, 3]
    smax  = Strnod[:, 4]
    
    s2=[]
        
    for i in range(npnod):
        s1.append( [ sa[i] , sb1[i] , sb2[i] ])
        s2.append( [ smin[i] , smax[i] , 0.0 ])
    
    elements=elements.astype(int)
    
    vtk1 = VtkData(\
            UnstructuredGrid(coordinates.tolist(),
                             line = elements.tolist()
                             ),
            PointData( Scalars(range(npnod)) ,
                       Vectors(du_xyz,name="Desplazamientos"),
                       Vectors(theta_xyz,name="Giros"),
                       Vectors(f_xyz,name="Force_Reaction"),
                       Vectors(m_xyz,name="Moment_Reaction"),
                       Vectors(s1,name="Tensiones"),
                       Vectors(s2,name="Tensiones max-min")
                       )
                )
    vtk1.tofile(file_name+'undeformed_')

    vtk2 = VtkData(\
            UnstructuredGrid(u_xyz,
                             line = elements.tolist()
                             ),
            PointData( Scalars(range(npnod)) ,
                       Vectors(du_xyz,name="Desplazamientos"),
                       Vectors(theta_xyz,name="Giros"),
                       Vectors(f_xyz,name="Force Reaction"),
                       Vectors(m_xyz,name="Moment Reaction"),
                       Vectors(s1,name="Tensiones"),
                       Vectors(s2,name="Tensiones max-min")
                       )
                )
    vtk2.tofile(file_name+'deformed_')
    
    print(".VTK file write success")

