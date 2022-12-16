using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, MLJ, Random, Distributions, MLJClusteringInterface, MLJMultivariateStatsInterface, CSV, StatsPlots, Serialization,  Distances

#preparing dataframes
test_input_PCA = deserialize("DATA/test_PCA.dat")
train_input_PCA = deserialize("DATA/train_PCA.dat")
train_data = CSV.read("DATA/train.csv", DataFrame)
coerce!(train_data, :labels => Multiclass) #needs coercing for confusion matrix
data_labels =  train_data.labels

#Kmeans on training set

Random.seed!(10); m1 = fit!(machine(KMeans(k = 6), train_input_PCA), verbosity = 1);
confusion_matrix(predict(m1, train_input_PCA), data_labels)
#better for CBP dont take seed : 8, 5



