# SSyDfem
This is a fem library in python for static and dynamic structural problems. The purpose of this library is for academic uses and teaching the finite element method.

It is based in other libraries avaibles in the web:

* CALFEM (https://github.com/CALFEM/calfem-python)
* MATFEM (http://www.cimne.com/mat-fem/plates.asp)

## Procedure

The procedure proposed to use this library is described with the next steps
* create the geometry file with gmsh. In this file we save the important entities with labely names. the extension of this file must be .vtk
You will need to download the software and learn the basics 
https://gmsh.info/

* create the python file and run it. Here the geometry is imported. The analysis is implemented and the results obtained.
* the results can be post process with a python code again or can be exported to see the displacements, tension or forces in gmsh o paraview.
Here you will find the paraview software
https://www.paraview.org/

## Installation

Install CALFEM for python using pip install 


## References

Rao, S.S.. (2017). The finite element method in engineering. 
Oñate, Eugenio. (2009). Structural Analysis with the Finite Element Method. Linear Statics. Vol. 1: Basis and Solids. 10.1007/978-1-4020-8733-2. 
Kausel, Eduardo. 2017. Advanced Structural Dynamics. Cambridge: Cambridge University Press.
Oñate, Eugenio. (2010). Structural analysis with the finite element method. Linear statics. Volume 2: Beams, plates and shells. 