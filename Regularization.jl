using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, MLJ, MLJLinearModels, MLCourse, Random, MLJDecisionTreeInterface, LinearAlgebra, CSV, Serialization

train_input = CSV.read("DATA/trainX.csv", DataFrame)
train_class = deserialize("DATA/trainlabels.dat")
test_input = CSV.read("DATA/testX.csv", DataFrame)
train_input_PCA = deserialize("DATA/train_PCA.dat")
test_input_PCA = deserialize("DATA/test_PCA.dat")
Y = categorical(train_class, levels = ["KAT5", "eGFP", "CBP"], ordered = true)

# linear model start 23h41 end 3h15

mach_lin = machine(LogisticClassifier(), train_input, Y)
fit!(mach_lin, verbosity = 2)
y_pred_lin = predict_mode(mach_lin, Y)
y_pred_lin_string = String.(y_pred_lin)
df_y_pred_lin = DataFrame(id = 1:3093, prediction = y_pred_lin_string)
CSV.write("RESULTS/predict_linear.csv", df_y_pred_lin, writeheader = true)

#linear model with PCA start 8h11 8h57

mach_lin_PCA = machine(LogisticClassifier(), train_input_PCA, Y)
fit!(mach_lin_PCA, verbosity = 2)
y_pred_lin_PCA = predict_mode(mach_lin_PCA, test_input_PCA)
y_pred_lin_string_PCA = String.(y_pred_lin_PCA)
df_y_pred_lin_PCA = DataFrame(id = 1:3093, prediction = y_pred_lin_string_PCA)
CSV.write("RESULTS/predict_linear_PCA.csv", df_y_pred_lin_PCA, writeheader = true)

#lasso -> standardized data ?

model_lasso = LogisticClassifier(penalty = :l1, lambda = 0.01)
self_tuning_model_lasso = TunedModel(model = model_lasso, resampling = CV(nfolds = 10), tuning = Grid(), range = range(model_lasso, :lambda, scale = :log10, lower = 1e-6, upper = 1e-2), measure = MisclassificationRate())
self_mach_lasso = machine(self_tuning_model_lasso, train_input, train_class)
fit!(self_mach_lasso, verbosity = 2)
y_pred_lasso = predict_mode(self_mach_lasso, test_input)
y_pred_lasso_string = String.(y_pred_lasso)
df_y_pred_lasso = DataFrame(id = 1:3093, prediction = y_pred_lasso_string)
CSV.write("RESULTS/predict_lasso.csv", df_y_pred_lasso, writeheader = true)
fitted_params(self_mach_lasso).best_model

# best lambda found by TunedModel is lambda = 0.00046415888336127773

#lasso with PCA start 14:09  # add random seed = 0
self_tuning_model_lasso_2 = TunedModel(model = model_lasso, resampling = CV(nfolds = 10), tuning = Grid(), range = range(model_lasso, :lambda, scale = :log10, lower = 1e-7, upper = 1e-5), measure = MisclassificationRate())
self_mach_lasso_PCA = machine(self_tuning_model_lasso_2, train_input_PCA, Y)
fit!(self_mach_lasso_PCA, verbosity = 2)
y_pred_lasso_PCA = predict_mode(self_mach_lasso_PCA, test_input_PCA)
y_pred_lasso_PCA_string = String.(y_pred_lasso_PCA)
df_y_pred_lasso_PCA = DataFrame(id = 1:3093, prediction = y_pred_lasso_PCA_string)
CSV.write("RESULTS/predict_lasso_PCA.csv", df_y_pred_lasso_PCA, writeheader = true)
fitted_params(self_mach_lasso_PCA).best_model

# best lambda found by TunedModel is 1e-6 with range 1e-6, 1e-2 and with range 1e-8, 1e-6 -> range 1e-7, 1e-5, lambda = 4.641588833612782e-7
#ran for about 1h30 instead of 6h30 for the model without PCA -> because GD often doesn't converge

#ridge
mach_ridge = machine(LogisticClassifier(penalty = :l2, lambda = 1), train_input, train_class)
fit!(mach_ridge)
y_pred_ridge = predict_mode(mach_ridge, test_input)
confusion_matrix(y_pred_ridge, test_class)
