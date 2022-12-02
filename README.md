# ML_chloelia

## Installation
1) Julia version 1.7.3 need to be pre-installed
2) MLcourse installation:
launch julia and run the following code to install the course material:

"julia> using Pkg
       Pkg.activate(temp = true)
       Pkg.develop(url = "https://github.com/jbrea/MLCourse")
       Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
       Pkg.instantiate()
       using MLCourse
       MLCourse.start()"
       
3) we recommend to use Visual Studio Code
4) 
## After first Git Pull
Data sets "test.csv" and "train.csv" need to be downloaded from: 
- https://lcnwww.epfl.ch/bio322/project2022/train.csv.gz 
- https://lcnwww.epfl.ch/bio322/project2022/test.csv.gz
and added into the empty folder "DATA" under the following paths: 
