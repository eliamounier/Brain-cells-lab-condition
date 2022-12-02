using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV
using MLCourse
using DataFrames
using Plots

#read the first 50 Rows of Test Data
train_data = CSV.read("DATA/train.csv", DataFrame, limit = 50)
train_data
# we can see on the terminal : Column = genes ; Rows = cell n° ==> we get the level of expression for each cell 
# train_data_dm = dropmissing(train_data) not needed, no missing Data