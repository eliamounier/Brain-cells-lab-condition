using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV
using MLCourse
using DataFrames
using Plots

#read the first 50 Rows of Test Data
test_data = CSV.read("DATA/test.csv", DataFrame, limit = 50)
test_data
# we can see on the terminal : Column = genes ; Rows = cell n° ==> we get the level of expression for each cell 