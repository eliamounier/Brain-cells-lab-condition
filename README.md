# ML_chloelia

## Installation
1) Julia version 1.7.3 need to be pre-installed
2) MLcourse installation:
launch julia and run the following code to install the course material:
julia> using Pkg
       Pkg.activate(temp = true)
       Pkg.develop(url = "https://github.com/jbrea/MLCourse")
       Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
       Pkg.instantiate()
       using MLCourse
       MLCourse.start()
