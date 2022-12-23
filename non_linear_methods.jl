# This folder is for non-linear methods
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, MLJ, MLCourse, Distributions, Serialization, MLJFlux, Flux, Random, MLJDecisionTreeInterface, CSV, MLJXGBoostInterface, Plots


#PART 1: no PCA
#preparing dataframes
test_input = CSV.read("DATA/testX.csv", DataFrame)
train_input = CSV.read("DATA/trainX.csv", DataFrame)
train_y = deserialize("DATA/trainlabels.dat")
train_input_PCA = deserialize("DATA/train_PCA.dat")
test_input_PCA = deserialize("DATA/test_PCA.dat")
Y = categorical(train_y, levels = ["KAT5", "eGFP", "CBP"], ordered = true)

#FOREST TREE
#machine for a random forest 
mach_forest_tree = machine(RandomForestClassifier(n_trees = 3500), train_input, Y) |> fit!
#prediction on the test set with random forest:
prediction_forest_tree = String.(predict_mode(mach_forest_tree, test_input))
df_predict_tree = DataFrame(id = 1:3093, prediction = prediction_forest_tree)

CSV.write("RESULTS/predict_foresttree_3500.csv", df_predict_tree)


#NEURON NETWORK CLASSIFIER 
#model 1
mach_neuron_network_classifier = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 128, dropout = 0.1, σ = relu),
                                                                batch_size = 32, epochs = 30),train_input, Y)|> fit!;

prediction_neural_network = String.(predict_mode(mach_neuron_network_classifier, test_input))
df_neural_network = DataFrame(id = 1:3093, prediction = prediction_neural_network)
CSV.write("RESULTS/neuralnetwork_1.csv", df_neural_network)
#model 2
mach_neuron_network_classifier2 = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, n_out))), batch_size = 32, epochs = 20),
train_input, Y)|> fit!;
prediction_neural_network2 = String.(predict_mode(mach_neuron_network_classifier2, test_input))
df_neural_network2 = DataFrame(id = 1:3093, prediction = prediction_neural_network2)
CSV.write("RESULTS/neuralnetwork_2.csv", df_neural_network2)
#model 3
mach_neuron_network_classifier3 = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 200, relu), Dense(200, n_out))), batch_size = 32, epochs = 30),
train_input, Y)|> fit!;
prediction_neural_network3 = String.(predict_mode(mach_neuron_network_classifier3, test_input))
df_neural_network3 = DataFrame(id = 1:3093, prediction = prediction_neural_network3)
CSV.write("RESULTS/neuralnetwork_3.csv", df_neural_network3)
#model 4 
mach_neuron_network_classifier4 = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 400, relu), Dense(400, n_out))), batch_size = 32, epochs = 30),
train_input, Y)|> fit!;
prediction_neural_network4 = String.(predict_mode(mach_neuron_network_classifier4, test_input))
df_neural_network4 = DataFrame(id = 1:3093, prediction = prediction_neural_network4)
CSV.write("RESULTS/neuralnetwork_4.csv", df_neural_network4)
#model 5
mach_neuron_network_classifier5 = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, epochs = 30),
train_input, Y)|> fit!;
prediction_neural_network5 = String.(predict_mode(mach_neuron_network_classifier5, test_input))
df_neural_network5 = DataFrame(id = 1:3093, prediction = prediction_neural_network5)
CSV.write("RESULTS/neuralnetwork_5.csv", df_neural_network5)


#REGULARISATION APPLIED TO NeuralNetworkClassifier
#L1 with lambda = 10e^-3
mach_neuron_network_classifier_L1_1= machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, lambda = 1e-3, alpha = 1, epochs = 30),
train_input, Y)|> fit!;
prediction_neural_network_L1_1 = String.(predict_mode(mach_neuron_network_classifier_L1_1, test_input))
df_neural_network_L1_1 = DataFrame(id = 1:3093, prediction = prediction_neural_network_L1_1)
CSV.write("RESULTS/neuralnetwork_L1_1.csv", df_neural_network_L1_1)

#L2 with lambda = 10e-3
mach_neuron_network_classifier_L2_1= machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, lambda = 1e-3, alpha = 0, epochs = 30),
train_input, Y)|> fit!;
prediction_neural_network_L2_1 = String.(predict_mode(mach_neuron_network_classifier_L2_1, test_input))
df_neural_network_L2_1 = DataFrame(id = 1:3093, prediction = prediction_neural_network_L2_1)
CSV.write("RESULTS/neuralnetwork_L2_1.csv", df_neural_network_L2_1)

#L2 3 bis to compare (without PCA)
mach_neuron_network_classifier_L2_2 = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 200, dropout = 0.01),
                                                                batch_size = 20, lambda = 1e-3, alpha = 0,  epochs = 40), train_input, Y)|> fit!; 
prediction_neural_network_L2_2= String.(predict_mode(mach_neuron_network_classifier_L2_2, test_input))
df_neural_network_L2_2 = DataFrame(id = 1:3093, prediction = prediction_neural_network_L2_2)
CSV.write("RESULTS/neuralnetwork_L2_3_bis.csv", df_neural_network_L2_2)


#PART 2: with PCA
test_input_PCA = deserialize("DATA/test_PCA.dat")
train_input_PCA = deserialize("DATA/train_PCA.dat")
train_y = deserialize("DATA/trainlabels.dat")
Y = categorical(train_y, levels = ["KAT5", "eGFP", "CBP"], ordered = true)

#GRABIENT BOOSTING TREES 
#would have taken multiple days to run
model_xgb = XGBoostClassifier()
r1 = range(model_xgb, :max_depth, lower = 2, upper = 5)
r2 = range(model_xgb, :eta, lower = 0.01, upper = 1)
r3 = range(model_xgb, :num_rounds, values = 100:100:1000)
self_tuning_model_xgb = TunedModel(model = model_xgb, resampling = CV(nfolds = 5), tuning = Grid(), range = [r1, r2, r3], measure = MisclassificationRate())
self_tuning_mach_xgb = machine(self_tuning_model_xgb, train_input_PCA, Y)
fit!(self_tuning_mach_xgb)
y_pred_xgb = predict_mode(self_tuning_mach_xgb, test_input_PCA)
y_pred_xgb_string = String.(y_pred_xgb)
df_y_pred_xgb = DataFrame(id = 1:3093, prediction = y_pred_xgb_string)
CSV.write("RESULTS/predict_xgb.csv", df_y_pred_xgb, writeheader = true)

# NEURON NETWORK
# preliminary tests
#model 1
mach_neuron_network_classifier_PCA = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 128, dropout = 0.1, σ = relu),
                                                                batch_size = 32, epochs = 30),train_input_PCA, Y)|> fit!;

prediction_neural_network_PCA = String.(predict_mode(mach_neuron_network_classifier_PCA, test_input_PCA))
df_neural_network_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network_PCA)
CSV.write("RESULTS/neuralnetwork_PCA_1.csv", df_neural_network_PCA)
#model 2
mach_neuron_network_classifier2_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, n_out))), batch_size = 32, epochs = 20),
train_input_PCA, Y)|> fit!;
prediction_neural_network2_PCA= String.(predict_mode(mach_neuron_network_classifier2_PCA, test_input_PCA))
df_neural_network2_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network2_PCA)
CSV.write("RESULTS/neuralnetwork_PCA_2.csv", df_neural_network2_PCA)
#model 3
mach_neuron_network_classifier3_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 200, relu), Dense(200, n_out))), batch_size = 32, epochs = 30),
train_input_PCA, Y)|> fit!;
prediction_neural_network3_PCA = String.(predict_mode(mach_neuron_network_classifier3_PCA, test_input_PCA))
df_neural_network3_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network3_PCA)
CSV.write("RESULTS/neuralnetwork_PCA_3.csv", df_neural_network3_PCA)
#model 4 
mach_neuron_network_classifier4_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 400, relu), Dense(400, n_out))), batch_size = 32, epochs = 30),
train_input_PCA, Y)|> fit!;
prediction_neural_network4_PCA = String.(predict_mode(mach_neuron_network_classifier4_PCA, test_input_PCA))
df_neural_network4_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network4_PCA)
CSV.write("RESULTS/neuralnetwork_PCA_4.csv", df_neural_network4_PCA)
#model 5
mach_neuron_network_classifier5_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, epochs = 30),
train_input_PCA, Y)|> fit!;
prediction_neural_network5_PCA = String.(predict_mode(mach_neuron_network_classifier5_PCA, test_input_PCA))
df_neural_network5_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network5_PCA)
CSV.write("RESULTS/neuralnetwork_PCA_5.csv", df_neural_network5_PCA)

#REGULARISATION APPLIED TO NeuralNetworkClassifier
#L1 with lambda = 10e^-3
mach_neuron_network_classifier_L1_1_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, lambda = 1e-3, alpha = 1, epochs = 30),
train_input_PCA, Y)|> fit!;
prediction_neural_network_L1_1_PCA = String.(predict_mode(mach_neuron_network_classifier_L1_1_PCA, test_input_PCA))
df_neural_network_L1_1_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network_L1_1_PCA)
CSV.write("RESULTS/neuralnetwork_L1_PCA_1.csv", df_neural_network_L1_1_PCA)

#L2 with lambda = 10e-3
mach_neuron_network_classifier_L2_1_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, lambda = 1e-3, alpha = 0, epochs = 30),
train_input_PCA, Y)|> fit!;
prediction_neural_network_L2_1_PCA = String.(predict_mode(mach_neuron_network_classifier_L2_1_PCA, test_input_PCA))
df_neural_network_L2_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network_L2_1_PCA)
CSV.write("RESULTS/neuralnetwork_L2_PCA_1.csv", df_neural_network_L2_PCA)


# TUNING NEURON CLASSIFIER
# multiple hyperparameters are all too long to tune with TunedModel. observations of oders of magnitude

#tuning of epochs by looking at the learning curve -> decided not to go higher to limit the running time for the tuning of dropout
Random.seed!(10)
model_neuron_classifier_L1_1_PCA_EP = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 256), batch_size = 32, lambda = 1e-6, alpha = 1, rng = 10)
mach_neuron_classifier_L1_1_PCA_EP = machine(model_neuron_classifier_L1_1_PCA_EP, train_input_PCA, Y)
curve = learning_curve(mach_neuron_classifier_L1_1_PCA_EP; range = range(model_neuron_classifier_L1_1_PCA_EP, :epochs, values = 10:100:1000), resampling = Holdout(), measure = MisclassificationRate())
curve_plotted = plot(curve.parameter_values, curve.measurements, xlab=curve.parameter_name, xscale=curve.parameter_scale, ylab = "Holdout estimate of MisclassificationRate")
png(curve_plotted, "PLOTS/learning_curve_NC_L1_PCA_epochs4.png")

# TunedModel on dropout 
# -> ran for 9h30 before we stopped it because of time constraints
Random.seed!(10)
model_neuron_classifier_L1_1_PCA_EP_DO = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 256), batch_size = 32, lambda = 1e-6, alpha = 1, rng = 10, epochs = 800)
r_dropout = range(model_neuron_classifier_L1_1_PCA_EP_DO, :(builder.dropout), values = [0.01, 0.1, 0.2, 0.3])
self_tuning_model_DO = TunedModel(model = model_neuron_classifier_L1_1_PCA_EP_DO, resampling = CV(nfolds = 6), tuning = Grid(), range = r_dropout, measure = MisclassificationRate())
self_mach_neuron_classifier_L1_1_PCA_EP_DO = machine(self_tuning_model_DO, train_input_PCA, Y)
fit!(self_mach_neuron_classifier_L1_1_PCA_EP_DO)
prediction_neural_classifier_L1_1_PCA_EP_DO = String.(predict_mode(self_mach_neuron_classifier_L1_1_PCA_EP_DO, test_input_PCA))
df_DO = DataFrame(id = 1:3093, prediction = prediction_neural_classifier_L1_1_PCA_EP_DO)
CSV.write("RESULTS/neuronclassifier_L1_PCA_DO.csv", df_DO)
fitted_params(self_mach_neuron_classifier_L1_1_PCA_EP_DO).best_model

#dropout but with confusion matrix
# tried with dropout = 
train_CM_X = train_input_PCA[1:3000, :]
test_CM_X = train_input_PCA[3001:5000, :]
train_CM_Y = Y[1:3000]
test_CM_Y = Y[3001:5000]

Random.seed!(10)
model_neuron_classifier_L1_1_PCA_CM = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 256, dropout = 0.3), batch_size = 32, lambda = 1e-6, alpha = 1, rng = 10, epochs = 800)
mach_neuron_classifier_L1_1_PCA_CM = machine(model_neuron_classifier_L1_1_PCA_CM, train_CM_X, train_CM_Y)
fit!(mach_neuron_classifier_L1_1_PCA_CM)
y_CM = predict_mode(mach_neuron_classifier_L1_1_PCA_CM, test_CM_X)
CM = confusion_matrix(y_CM, test_CM_Y)

Random.seed!(10)
model_neuron_classifier_L1_1_PCA_CM2 = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 256, dropout = 0.1), batch_size = 32, lambda = 1e-6, alpha = 1, rng = 10, epochs = 800)
mach_neuron_classifier_L1_1_PCA_CM2 = machine(model_neuron_classifier_L1_1_PCA_CM2, train_CM_X, train_CM_Y)
fit!(mach_neuron_classifier_L1_1_PCA_CM2)
y_CM2 = predict_mode(mach_neuron_classifier_L1_1_PCA_CM2, test_CM_X)
CM2 = confusion_matrix(y_CM2, test_CM_Y)
#this confusion matrix gave the best MisclassificationRate

Random.seed!(10)
model_neuron_classifier_L1_1_PCA_CM3 = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 256, dropout = 0.01), batch_size = 32, lambda = 1e-6, alpha = 1, rng = 10, epochs = 800)
mach_neuron_classifier_L1_1_PCA_CM3 = machine(model_neuron_classifier_L1_1_PCA_CM3, train_CM_X, train_CM_Y)
fit!(mach_neuron_classifier_L1_1_PCA_CM3)
y_CM3 = predict_mode(mach_neuron_classifier_L1_1_PCA_CM3, test_CM_X)
CM3 = confusion_matrix(y_CM3, test_CM_Y)

# FINAL MODEL
## same as the second confusion matrix but with the whole training set
Random.seed!(10)
model_neuron_classifier_final = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 256, dropout = 0.1), batch_size = 32, lambda = 1e-6, alpha = 1, rng = 10, epochs = 800)
mach_neuron_classifier_final = machine(model_neuron_classifier_final, train_input_PCA, Y)
fit!(mach_neuron_classifier_final)
y_final = String.(predict_mode(mach_neuron_classifier_final, test_input_PCA))
df_final = DataFrame(id = 1:3093, prediction = y_final)
CSV.write("RESULTS/final_Nonlinear_L1_PCA.csv", df_final)