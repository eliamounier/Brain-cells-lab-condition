# This folder is for non-linear methods
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, CSV

train_input = CSV.read("DATA/dataX.csv", DataFrame)[1:100, :]
train_class = CSV.read("DATA/datay.csv", DataFrame)[1:100, :]
test_input = CSV.read("DATA/test.csv", DataFrame)

#FOREST TREE CLASSIFIER
using MLJDecisionTreeInterface
#machine for a random forest with 500 trees
mach_forest_tree = machine(RandomForestClassifier(n_trees = 50), train_input, train_class) |> fit!;
#prediction on the test set with random forest:
prediction_forest_tree = String.(predict_mode(mach_forest_tree, test_input))
df_predict_tree = Dataframe(id :: 1:3093, prediction_forest_tree)
CSV.write("./predict_foresttree.csv", prediction_forest_tree)

#accuracy = mean(prediction_forest_tree .== test_output)
#confusion_matrix(prediction_forest_tree, test_output)


#NEURON NETWORK CLASSIFIER
using MLJ, MLJFlux, Flux
#machine for neuron network classifier
mach_neuron_network = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 128, dropout = 0.1, σ = relu),
                                                                batch_size = 32, epochs = 30),train_input, train_class)|> fit!;
#prediction on test set
prediction_neuron_network = String.(predict_mode(mach_neuron_network, test_input))

df_neuron_network = Dataframe(id :: 1:3093, prediction_neuron_network)
CSV.write("./predict_neuronnetwork.csv", prediction_neuron_network)

#IMPROVEMENT OF NEURON NETWORK CLASSIFIER WITH MULTILAYER PERCEPTRON




#CLUSTERING ?==> or for visualisation
using MLJ, MLJClusteringInterface, MLJMultivariateStatsInterface, Random

##K-means clustering
#machine with k = 3 clusters :
#Random.seed!(10); mach_k_means_clusters = fit!(machine(KMeans(k = 3), train_input));
#prediction with k-means clustering on training data
#prediction_k_means_clusters = MLJ.predict(mach_k_means_clusters, train_imput);