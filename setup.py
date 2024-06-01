from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()
    
VERSION = '0.0.3'
DESCRIPTION = 'Structural FEM library building in python.'


# Setting up
setup(
    name="SSyDfem",
    version=VERSION,
    author="nf-project (Fernando E. Burgos)",
    author_email="<burgosnfer@gmail.com>",
    description=DESCRIPTION,
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description_content_type="text/markdown",
    long_description=long_description,
    install_requires=['vtk',"PyVTK",'numpy','scipy','matplotlib','gmsh'],
    keywords=['python', 'fem'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)