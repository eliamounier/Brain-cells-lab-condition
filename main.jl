include("./myfunctions.jl")
include("./Data_treatment.jl")
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, OpenML, DataFrames, CSV, MLCourse, Random, Statistics, MLJ, MLJLinearModels, Distributions


#X = select(train_data, Not([:labels]))
#y = train_data.labels


# logistic classification
#model_LC = LogisticClassifier()
#mach_LC = machine(model_LC, X, y)
#fit!(mach_LC)

#forest tree (10-other non linear methods)

