using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, MLJ, MLJLinearModels, MLCourse, Random, MLJDecisionTreeInterface, LinearAlgebra, CSV, Serialization
using Distributions, MLJFlux, Flux, MLJXGBoostInterface, Plots

# loading the data
# (need to run Data_treatment.jl and PCA.jl beforehand)
train_class = deserialize("DATA/trainlabels.dat")
train_input_PCA = deserialize("DATA/train_PCA.dat")
test_input_PCA = deserialize("DATA/test_PCA.dat")
Y = categorical(train_class, levels = ["KAT5", "eGFP", "CBP"], ordered = true)


# Final linear model
model_lasso = LogisticClassifier(penalty = :l1, lambda = 0.01)
Random.seed!(10)
self_tuning_model_lasso_2 = TunedModel(model = model_lasso, resampling = CV(nfolds = 10), tuning = Grid(), range = range(model_lasso, :lambda, scale = :log10, lower = 1e-7, upper = 1e-5), measure = MisclassificationRate())
Random.seed!(10)
self_mach_lasso_PCA = machine(self_tuning_model_lasso_2, train_input_PCA, Y)
fit!(self_mach_lasso_PCA, verbosity = 2)
y_pred_lasso_PCA = predict_mode(self_mach_lasso_PCA, test_input_PCA)
y_pred_lasso_PCA_string = String.(y_pred_lasso_PCA)
df_y_pred_lasso_PCA = DataFrame(id = 1:3093, prediction = y_pred_lasso_PCA_string)
CSV.write("RESULTS/final_linear_L1_PCA.csv", df_y_pred_lasso_PCA, writeheader = true)
fitted_params(self_mach_lasso_PCA).best_model

# Final non-linear model
# see the hyperparameter testing in file non_linear_methods.jl
Random.seed!(10)
model_neuron_classifier_final = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 256, dropout = 0.1), batch_size = 32, lambda = 1e-6, alpha = 1, rng = 10, epochs = 800)
mach_neuron_classifier_final = machine(model_neuron_classifier_final, train_input_PCA, Y)
fit!(mach_neuron_classifier_final)
y_final = String.(predict_mode(mach_neuron_classifier_final, test_input_PCA))
df_final = DataFrame(id = 1:3093, prediction = y_final)
CSV.write("RESULTS/final_Nonlinear_L1_PCA.csv", df_final)
