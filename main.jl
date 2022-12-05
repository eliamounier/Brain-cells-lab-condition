include("./myfunctions.jl")
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, OpenML, DataFrames, CSV, MLCourse, Random, Statistics, MLJ, MLJLinearModels, Distributions

train_data = CSV.read("DATA/train.csv", DataFrame)
coerce!(train_data, :labels => Multiclass)

X = select(train_data, Not([:labels]))
y = train_data.labels
ilmioio

# logistic classification
model_LC = LogisticClassifier()
mach_LC = machine(model_LC, X, y)
fit!(mach_LC)