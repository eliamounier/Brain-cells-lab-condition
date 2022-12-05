# This folder is for non-linear methods
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, MLJ, MLJLinearModels, MLCourse, Random, Distributions, Plots, MLJFlux, Flux, OpenML, MLJDecisionTreeInterface, CSV

#importation of training data and test data:
train_data = CSV.read("DATA/train.csv", DataFrame)
coerce!(train_data, :labels => Multiclass)
test_data = CSV.read("DATA/test.csv", DataFrame)

train_input = select(train_data[1:100, :], Not([:labels]))
train_class = train_data.labels[1:100, :]

#FOREST TREE
#machine for a random forest with 500 trees
mach_forest_tree = machine(RandomForestClassifier(n_trees = 500), train_input, train_class) |> fit!;
#prediction on the test set with random forest:
prediction_forest_tree = predict_mode(mach_forest_tree, test_data)
prediction_forest_tree

#NEURON NETWORK CLASSIFIER
mach_neuron_network_classifier = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 128, dropout = 0.1, σ = relu),
                                                                batch_size = 32, epochs = 30),train_input, train_class)|> fit!;