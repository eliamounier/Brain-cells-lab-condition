# This folder is for non-linear methods
include("./Data_treatment.jl")
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, MLJ, MLJLinearModels, MLCourse, Random, Distributions, Plots, MLJFlux, Flux, OpenML, MLJDecisionTreeInterface, CSV

train_input = dataX
train_class = datay

test_input = test_data

#FOREST TREE
#machine for a random forest with 500 trees
mach_forest_tree = machine(RandomForestClassifier(n_trees = 750), train_input, train_class) |> fit!
#prediction on the test set with random forest:
prediction_forest_tree = predict_mode(mach_forest_tree, test_input)
prediction_forest_tree

accuracy = mean(prediction_forest_tree .== test_output)
confusion_matrix(prediction_forest_tree, test_output)

CSV.write("./predict_foresttree.csv", prediction_forest_tree)

#NEURON NETWORK CLASSIFIER
mach_neuron_network_classifier = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 128, dropout = 0.1, σ = relu),
                                                                batch_size = 32, epochs = 30),train_input, train_class)|> fit!;