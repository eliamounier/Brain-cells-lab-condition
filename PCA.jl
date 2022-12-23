using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, MLJ, DataFrames, MLJMultivariateStatsInterface, OpenML, Plots, LinearAlgebra, Statistics, Random, CSV, MLJLinearModels, Serialization
using Distributions, MLJClusteringInterface, StatsPlots

#importation of data sets
train_data_treated = CSV.read("DATA/trainX.csv", DataFrame)
test_data = CSV.read("DATA/testX.csv", DataFrame)
Y = deserialize("DATA/trainlabels.dat")
true_labels = categorical(Y, levels = ["KAT5", "eGFP", "CBP"], ordered = true)

#PCA on our treated data sets
pca_bis = machine(@pipeline(Standardizer(), PCA()), train_data_treated) |> fit!;
dataX = MLJ.transform(pca_bis, train_data_treated)
dataT = MLJ.transform(pca_bis, test_data)
serialize("DATA/train_PCA.dat", dataX) #saving new dataframes
serialize("DATA/test_PCA.dat", dataT)

#PLOTS for PCA

#PLOT 1 : PCA not standardized, in 2D, true labels 
data2d = MLJ.transform(fit!(machine(PCA(maxoutdim = 2), train_data_treated)), train_data_treated)
gr();
cl1_true = scatter(data2d.x1, data2d.x2, c = Int.(int(true_labels)), legend = false, title = "truelabels, PCA2D" )
png(cl1_true, "PLOTS/PCA_2D.png")

#PLOT 2 : same but standardized
data2d_SD = MLJ.transform(fit!(machine(@pipeline(Standardizer(),PCA(maxoutdim = 2)), train_data_treated)), train_data_treated)
cl1_true_SD = scatter(data2d_SD.x1, data2d_SD.x2, c = Int.(int(true_labels)), legend = false, title = "truelabels, PCA2D & Standardized" )
png(cl1_true_SD, "PLOTS/PCA_2D_sd.png")
# clusters are better represented (see plottings) : we will keep on using standardized PCA

#PLOT 3 : TNse, to see if clusters are better in this type of dimensional reduction
using TSne
train_data = MLJ.transform(fit!(machine(Standardizer(), train_data_treated),verbosity = 0)) # standardizing data
tsne_proj = tsne(Array(train_data), 2, 0, 2000, 50.0, progress = false);
TSne_train = scatter(tsne_proj[:, 1], tsne_proj[:, 2],c = Int.(int(true_labels)), xlabel = "TSne 1", ylabel = "TSne 2", legend = false, title = "TSne")
png(TSne_train, "PLOTS/TNse.png")
# as the plot suggests : TSne is not a good idea to identify clusters

#PLOT 4 : Explained variance for standardized-PCA
vars = report(pca_bis).pca.principalvars ./ report(pca_bis).pca.tprincipalvar
    p1 = plot(vars, label = nothing, yscale = :log10,
              xlabel = "component", ylabel = "proportion of variance explained")
    p2 = plot(cumsum(vars),
              label = nothing, xlabel = "component",
              ylabel = "cumulative prop. of variance explained")
    p_var = plot(p1, p2, layout = (1, 2), size = (700, 400))
png(p_var, "PLOTS/Pvar_explained")

#PLOT 5: BIPLOT: PCA-standardized, PC1 and PC2
gr(); #gr() modulus 
bp1_bis = biplot(pca_bis) 
png(bp1_bis, "PLOTS/Biplot.png") 

