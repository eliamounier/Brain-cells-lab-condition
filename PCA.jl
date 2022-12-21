using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, MLJ, DataFrames, MLJMultivariateStatsInterface, OpenML, Plots, LinearAlgebra, Statistics, Random, CSV, MLJLinearModels
import PlutoPlotly as PP
chop(x, eps = 1e-12) = abs(x) > eps ? x : zero(typeof(x))

train_data_treated = CSV.read("DATA/trainX.csv", DataFrame)

#PCA 
pca = fit!(machine(PCA(variance_ratio = 1), train_data_treated), verbosity = 1);

#biplot for PCA
gr(); #gr() modulus
bp1 = biplot(pca)
png(bp1, "PLOTS/Biplot_PCA_1.png")
bp2 = biplot(pca, pc = (1, 3))
png(bp2, "PLOTS/Biplot_PCA_2.png")



# CHLOE

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, MLJ, DataFrames, MLJMultivariateStatsInterface, LinearAlgebra, Statistics, Random, CSV, MLJLinearModels, Serialization
import PlutoPlotly as PP
chop(x, eps = 1e-12) = abs(x) > eps ? x : zero(typeof(x))

#importation of data sets
train_data_treated = CSV.read("DATA/trainX.csv", DataFrame)
test_data = CSV.read("DATA/testX.csv", DataFrame)

# Standardization + PCA on data sets
pca_bis = machine(@pipeline(Standardizer(), PCA()), train_data_treated) |> fit!;
dataX = MLJ.transform(pca_bis, train_data_treated)
dataT = MLJ.transform(pca_bis, test_data)
serialize("DATA/train_PCA.dat", dataX) #saving new dataframes
serialize("DATA/test_PCA.dat", dataT)
