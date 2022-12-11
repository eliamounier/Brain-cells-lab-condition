# This folder is for non-linear methods
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, MLJ, MLJLinearModels, MLCourse, Random, Distributions, Serialization, MLJFlux, Flux, OpenML, MLJDecisionTreeInterface, CSV

#preparing dataframes
test_input = CSV.read("DATA/testX.csv", DataFrame)
train_input = CSV.read("DATA/trainX.csv", DataFrame)
train_y = deserialize("DATA/trainlabels.dat")
Y = categorical(train_y, levels = ["KAT5", "eGFP", "CBP"], ordered = true)

#FOREST TREE
#machine for a random forest 
mach_forest_tree = machine(RandomForestClassifier(n_trees = 3500), train_input, Y) |> fit!
#prediction on the test set with random forest:
prediction_forest_tree = String.(predict_mode(mach_forest_tree, test_input))
df_predict_tree = DataFrame(id = 1:3093, prediction = prediction_forest_tree)

CSV.write("./predict_foresttree_3500.csv", df_predict_tree)

#Gradient boosting trees


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
training_loss = cross_entropy(predict(mach_neuron_network_classifier3, train_input), soe1) |> mean #very low => OVERFITTING!!!
#model 4 
function NeuralNetwork()
    return Chain(
            Dense(2, 25,relu),
            Dense(25,1,x->σ.(x))
            )
end

nn = Chain(Dense(1 100, relu), Dense(100, 1))