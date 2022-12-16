using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, MLJ, MLJLinearModels, MLCourse, MLJFlux, Flux, OpenML, MLJDecisionTreeInterface, CSV, NearestNeighborModels, Statistics, Random
include("./Data_treatment.jl")

# train_input = CSV.read("DATA/dataX.csv", DataFrame)
# train_class = CSV.read("DATA/datay.csv", DataFrame)

train_input = dataX
train_class = datay
test_input = dataT

# polynomial classification
model_pn = Polynomial() |> LogisticClassifier()
self_tuning_model_pn = TunedModel(model = model_pn, resampling = CV(nfolds = 10), range = range(model_pn, :(polynomial.degree), values = 1:10), measure = LogLoss())
self_tuning_mach_pn = machine(self_tuning_model_pn, train_input, train_class)
fit!(self_tuning_mach_pn, verbosity = 2)
report(self_tuning_mach_pn)
y_predict_pn = predict_mode(self_tuning_mach_pn, test_input)
y_predict_pn_string = String.(y_predict_pn)
df_y_predict_pn = DataFrame(id = 1:3093, prediction = y_predict_pn_string)
CSV.write("RESULTS/predict_pn.csv", df_y_predict_pn, writeheader = true)

#KNN
model_KNN = KNNClassifier()
self_tuning_model_KNN = TunedModel(model = model_KNN, resampling = CV(nfolds = 10), tuning = Grid(), range = range(model_KNN, :K, values = 5:50), measure = LogLoss())
self_tuning_machine_KNN = machine(self_tuning_model_KNN, train_input, train_class)
fit!(self_tuning_machine_KNN, verbosity = 2)
report(self_tuning_machine_KNN)
y_predict_KNN = predict_mode(self_tuning_machine_KNN, test_input)
y_predict_KNN_string = String.(y_predict_KNN)
df_y_predict_KNN = DataFrame(id = 1:3093, prediction = y_predict_KNN_string)
CSV.write("RESULTS/predict_pn.csv", df_y_predict_KNN, writeheader = true)
