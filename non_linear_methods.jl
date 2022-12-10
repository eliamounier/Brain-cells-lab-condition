# This folder is for non-linear methods
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, MLJ, MLJLinearModels, MLCourse, Random, Distributions, Plots, MLJFlux, Flux, OpenML, MLJDecisionTreeInterface, CSV


train_input = CSV.read("DATA/trainX.csv", DataFrame)
train_class = CSV.read("DATA/trainy.csv", DataFrame)

using Serialization
soe = deserialize("DATA/trainlabels.dat")
soe1 = categorical(soe, levels = ["KAT5", "eGFP", "CBP"], ordered = true)

test_input = CSV.read("DATA/testX.csv", DataFrame)

#FOREST TREE
#machine for a random forest with 500 trees
mach_forest_tree = machine(RandomForestClassifier(n_trees = 3000), train_input, soe1) |> fit!
#prediction on the test set with random forest:
prediction_forest_tree = String.(predict_mode(mach_forest_tree, test_input))
df_predict_tree = DataFrame(id = 1:3093, prediction = prediction_forest_tree)


accuracy = mean(prediction_forest_tree .== test_output)
confusion_matrix(prediction_forest_tree, test_output)


CSV.write("./predict_foresttree_3000.csv", df_predict_tree)

#NEURON NETWORK CLASSIFIER
mach_neuron_network_classifier = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 128, dropout = 0.1, σ = relu),
                                                                batch_size = 32, epochs = 30),train_input, train_class)|> fit!;