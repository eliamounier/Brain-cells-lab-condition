# This folder is for non-linear methods
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, MLJ, MLCourse, Distributions, Serialization, MLJFlux, Flux, OpenML, MLJDecisionTreeInterface, CSV, MLJXGBoostInterface


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

CSV.write("./predict_foresttree_3500.csv", df_predict_tree)


#Gradient boosting trees -> to do with PCA
model_xgb = XGBoostClassifier(eta = 0.1, num_round = 1000)
self_tuning_model_xgb = TunedModel(model = model_xgb, resampling = CV(nfolds = 5), tuning = Grid(), range = range(model_xgb, :max_depth, lower = 3, upper = 5), measure = MisclassificationRate())
self_tuning_mach_xgb = machine(self_tuning_model_xgb, train_input_PCA, Y)
fit!(self_tuning_mach_xgb)
y_pred_xgb = predict_mode(self_tuning_mach_xgb, test_input_PCA)
y_pred_xgb_string = String.(y_pred_xgb)
df_y_pred_xgb = DataFrame(id = 1:3093, prediction = y_pred_xgb_string)
CSV.write("RESULTS/predict_xgb.csv", df_y_pred_xgb, writeheader = true)


#NEURON NETWORK CLASSIFIER 
#model 1
mach_neuron_network_classifier = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 128, dropout = 0.1, σ = relu),
                                                                batch_size = 32, epochs = 30),train_input, Y)|> fit!;

prediction_neural_network = String.(predict_mode(mach_neuron_network_classifier, test_input))
df_neural_network = DataFrame(id = 1:3093, prediction = prediction_neural_network)
CSV.write("./neuralnetwork_1.csv", df_neural_network)
#model 2
mach_neuron_network_classifier2 = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, n_out))), batch_size = 32, epochs = 20),
train_input, Y)|> fit!;
prediction_neural_network2 = String.(predict_mode(mach_neuron_network_classifier2, test_input))
df_neural_network2 = DataFrame(id = 1:3093, prediction = prediction_neural_network2)
CSV.write("./neuralnetwork_2.csv", df_neural_network2)
#model 3
mach_neuron_network_classifier3 = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 200, relu), Dense(200, n_out))), batch_size = 32, epochs = 30),
train_input, Y)|> fit!;
prediction_neural_network3 = String.(predict_mode(mach_neuron_network_classifier3, test_input))
df_neural_network3 = DataFrame(id = 1:3093, prediction = prediction_neural_network3)
CSV.write("./neuralnetwork_3.csv", df_neural_network3)
#model 4 
mach_neuron_network_classifier4 = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 400, relu), Dense(400, n_out))), batch_size = 32, epochs = 30),
train_input, Y)|> fit!;
prediction_neural_network4 = String.(predict_mode(mach_neuron_network_classifier4, test_input))
df_neural_network4 = DataFrame(id = 1:3093, prediction = prediction_neural_network4)
CSV.write("./neuralnetwork_4.csv", df_neural_network4)
#model 5
mach_neuron_network_classifier5 = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, epochs = 30),
train_input, Y)|> fit!;
prediction_neural_network5 = String.(predict_mode(mach_neuron_network_classifier5, test_input))
df_neural_network5 = DataFrame(id = 1:3093, prediction = prediction_neural_network5)
CSV.write("./neuralnetwork_5.csv", df_neural_network5)


#REGULARISATION APPLIED TO NeuralNetworkClassifier
#L1 with lambda = 10e^-3
mach_neuron_network_classifier_L1_1= machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, lambda = 1e-3, alpha = 1, epochs = 30),
train_input, Y)|> fit!;
prediction_neural_network_L1_1 = String.(predict_mode(mach_neuron_network_classifier_L1_1, test_input))
df_neural_network_L1_1 = DataFrame(id = 1:3093, prediction = prediction_neural_network_L1_1)
CSV.write("./neuralnetwork_L1_1.csv", df_neural_network_L1_1)

#L2 with lambda = 10e-3
mach_neuron_network_classifier_L2_1= machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, lambda = 1e-3, alpha = 0, epochs = 30),
train_input, Y)|> fit!;
prediction_neural_network_L2_1 = String.(predict_mode(mach_neuron_network_classifier_L2_1, test_input))
df_neural_network_L2_1 = DataFrame(id = 1:3093, prediction = prediction_neural_network_L2_1)
CSV.write("./neuralnetwork_L2_1.csv", df_neural_network_L2_1)

#L2 3 bis to compare (without PCA)
mach_neuron_network_classifier_L2_2 = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 200, dropout = 0.01),
                                                                batch_size = 20, lambda = 1e-3, alpha = 0,  epochs = 40), train_input, Y)|> fit!; 
prediction_neural_network_L2_2= String.(predict_mode(mach_neuron_network_classifier_L2_2, test_input))
df_neural_network_L2_2 = DataFrame(id = 1:3093, prediction = prediction_neural_network_L2_2)
CSV.write("./neuralnetwork_L2_3_bis.csv", df_neural_network_L2_2)


#PART 2: with PCA
test_input_PCA = deserialize("DATA/test_PCA.dat")
train_input_PCA = deserialize("DATA/train_PCA.dat")
train_y = deserialize("DATA/trainlabels.dat")
Y = categorical(train_y, levels = ["KAT5", "eGFP", "CBP"], ordered = true)

#model 1
mach_neuron_network_classifier_PCA = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 128, dropout = 0.1, σ = relu),
                                                                batch_size = 32, epochs = 30),train_input_PCA, Y)|> fit!;

prediction_neural_network_PCA = String.(predict_mode(mach_neuron_network_classifier_PCA, test_input_PCA))
df_neural_network_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network_PCA)
CSV.write("./neuralnetwork_PCA_1.csv", df_neural_network_PCA)
#model 2
mach_neuron_network_classifier2_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, n_out))), batch_size = 32, epochs = 20),
train_input_PCA, Y)|> fit!;
prediction_neural_network2_PCA= String.(predict_mode(mach_neuron_network_classifier2_PCA, test_input_PCA))
df_neural_network2_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network2_PCA)
CSV.write("./neuralnetwork_PCA_2.csv", df_neural_network2_PCA)
#model 3
mach_neuron_network_classifier3_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 200, relu), Dense(200, n_out))), batch_size = 32, epochs = 30),
train_input_PCA, Y)|> fit!;
prediction_neural_network3_PCA = String.(predict_mode(mach_neuron_network_classifier3_PCA, test_input_PCA))
df_neural_network3_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network3_PCA)
CSV.write("./neuralnetwork_PCA_3.csv", df_neural_network3_PCA)
#model 4 
mach_neuron_network_classifier4_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 400, relu), Dense(400, n_out))), batch_size = 32, epochs = 30),
train_input_PCA, Y)|> fit!;
prediction_neural_network4_PCA = String.(predict_mode(mach_neuron_network_classifier4_PCA, test_input_PCA))
df_neural_network4_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network4_PCA)
CSV.write("./neuralnetwork_PCA_4.csv", df_neural_network4_PCA)
#model 5
mach_neuron_network_classifier5_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, epochs = 30),
train_input_PCA, Y)|> fit!;
prediction_neural_network5_PCA = String.(predict_mode(mach_neuron_network_classifier5_PCA, test_input_PCA))
df_neural_network5_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network5_PCA)
CSV.write("./neuralnetwork_PCA_5.csv", df_neural_network5_PCA)

#REGULARISATION APPLIED TO NeuralNetworkClassifier
#L1 with lambda = 10e^-3
mach_neuron_network_classifier_L1_1_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, lambda = 1e-3, alpha = 1, epochs = 30),
train_input_PCA, Y)|> fit!;
prediction_neural_network_L1_1_PCA = String.(predict_mode(mach_neuron_network_classifier_L1_1_PCA, test_input_PCA))
df_neural_network_L1_1_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network_L1_1_PCA)
CSV.write("./neuralnetwork_L1_PCA_1.csv", df_neural_network_L1_1_PCA)

#L2 with lambda = 10e-3
mach_neuron_network_classifier_L2_1_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, lambda = 1e-3, alpha = 0, epochs = 30),
train_input_PCA, Y)|> fit!;
prediction_neural_network_L2_1_PCA = String.(predict_mode(mach_neuron_network_classifier_L2_1_PCA, test_input_PCA))
df_neural_network_L2_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network_L2_1_PCA)
CSV.write("./neuralnetwork_L2_PCA_1.csv", df_neural_network_L2_PCA)

#training with a linear model /gradient descent
model = NeuralNetworkClassifier(builder = MLJFlux.Linear(σ = identity), epochs = 100, batch_size = 128)
mach = machine(model, train_input_PCA, Y)
evaluate!(mach, resampling = CV(nfolds = 5, shuffle = true), force = true, repeats = 3, measure = [mcr, log_loss])

########
#L2 bis : more epoch 
mach_neuron_network_classifier_L2_2_PCA = machine(NeuralNetworkClassifier( builder = MLJFlux.@builder(Chain(Dense(n_in, 128, relu), Dense(128, n_out))), batch_size = 32, lambda = 1e-3, alpha = 0, epochs = 60),
train_input_PCA, Y)|> fit!;
prediction_neural_network_L2_2_PCA = String.(predict_mode(mach_neuron_network_classifier_L2_2_PCA, test_input_PCA))
df_neural_network_L2_2_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network_L2_2_PCA)
CSV.write("./neuralnetwork_L2_PCA_2_bis2.csv", df_neural_network_L2_2_PCA)

#L2.3bis
mach_neuron_network_classifier_PCA = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 200, dropout = 0.01),
                                                                batch_size = 20, lambda = 1e-3, alpha = 0,  epochs = 40),train_input_PCA, Y)|> fit!; 

prediction_neural_network_PCA = String.(predict_mode(mach_neuron_network_classifier_PCA, test_input_PCA))
df_neural_network_PCA = DataFrame(id = 1:3093, prediction = prediction_neural_network_PCA)
CSV.write("./neuralnetwork_L2_PCA_3_bis.csv", df_neural_network_PCA)