using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, MLJ, MLJLinearModels, MLCourse, Random, Distributions, Plots, MLJFlux, Flux, OpenML, MLJDecisionTreeInterface, LinearAlgebra, CSV
include("./Data_treatment.jl")

# train_input = CSV.read("DATA/dataX.csv", DataFrame)
# train_class = CSV.read("DATA/datay.csv", DataFrame)

train_input = dataX
train_class = datay
#test_input = CSV.read("DATA/dataT.csv", DataFrame)

test_input = dataT

#lasso
mach_lasso = machine(LogisticClassifier(penalty = :l1, lambda = 1e-3), train_input, train_class)
fit!(mach_lasso)
y_pred = predict_mode(mach_lasso, test_input)
confusion_matrix(y_pred, test_class)

model_lasso = LogisticClassifier(penalty = :l1)
self_tuning_model_lasso = TunedModel(model = model_lasso, resampling = CV(), tuning = Grid(), range = range(model_lasso, :lambda, scale = :log10, lower = 1e-8, upper = 1e-1), measure = MisclassificationRate())
self_mach_lasso = machine(self_tuning_model_lasso, train_input, train_class)
fit!(self_mach_lasso, verbosity = 2)

#ridge
mach_ridge = machine(LogisticClassifier(penalty = :l2, lambda = 1), train_input, train_class)
fit!(mach_ridge)
y_pred_ridge = predict_mode(mach_ridge, test_input)
confusion_matrix(y_pred_ridge, test_class)

LogLoss()