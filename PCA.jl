using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, MLJ, DataFrames, MLJMultivariateStatsInterface, OpenML, Plots, LinearAlgebra, Statistics, Random, CSV, MLJLinearModels, Serialization
import PlutoPlotly as PP
chop(x, eps = 1e-12) = abs(x) > eps ? x : zero(typeof(x))

#importation of data sets
train_data_treated = CSV.read("DATA/trainX.csv", DataFrame)
test_data = CSV.read("DATA/testX.csv", DataFrame)

#PCA 
pca_bis = machine(@pipeline(Standardizer(), PCA()), train_data_treated) |> fit!;
dataX = MLJ.transform(pca_bis, train_data_treated)
dataT = MLJ.transform(pca_bis, test_data)
serialize("DATA/train_PCA.dat", dataX) #saving new dataframes
serialize("DATA/test_PCA.dat", dataT)


#PLOTS for PCA
using Distributions, MLJClusteringInterface, StatsPlots, Serialization 
Y = deserialize("DATA/trainlabels.dat")
true_labels = categorical(Y, levels = ["KAT5", "eGFP", "CBP"], ordered = true)

#PLOT 1 : PCA not standardized, in 2D, true labels 
data2d = MLJ.transform(fit!(machine(PCA(maxoutdim = 2), train_data_treated)), train_data_treated)
gr()
cl1_true = scatter(data2d.x1, data2d.x2, c = Int.(int(true_labels)), legend = false, title = "truelabels, PCA2D" )
png(cl1_true, "PLOTS/PCA_2D.png")

#PLOT 2 : same but standardized
data2d_SD = MLJ.transform(fit!(machine(@pipeline(Standardizer(),PCA(maxoutdim = 2)), train_data_treated)), train_data_treated)
cl1_true_SD = scatter(data2d_SD.x1, data2d_SD.x2, c = Int.(int(true_labels)), legend = false, title = "truelabels, PCA2D & Standardized" )
png(cl1_true_SD, "PLOTS/PCA_2D_sd.png")
# clusters are better represented : we will keep on using standardized PCA

#PLOT 3 : TNse, to see if clusters are better in this type of dimensional reduction
using TSne
train_data = MLJ.transform(fit!(machine(Standardizer(), train_data_treated),verbosity = 0)) # standardizing data
tsne_proj = tsne(Array(train_data), 2, 0, 2000, 50.0, progress = false);
TSne_train = scatter(tsne_proj[:, 1], tsne_proj[:, 2],c = Int.(int(true_labels)), xlabel = "TSne 1", ylabel = "TSne 2", legend = false, title = "TSne")
png(TSne_train, "PLOTS/TNse.png")
# as the plot suggests : TSne is not a good idea to identify clusters

#PLOT 4: Explaned variance for standardized-PCA
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
png(bp1_bis, "PLOTS/Biplot_PCA_standardized_1.png") 



#######TO REMOVE ###### (just so that it is saved somewhere )
#PCA biplot 20dimension
pca20D = fit!(machine(@pipeline(Standardizer(), PCA(maxoutdim = 20)), train_data_treated))
gr(); 
bpPCA20D = biplot(pca20D) 
png(bpPCA20D, "PLOTS/Biplot_PCA_20D.png") 
report(pca20D)


#PCA biplot 20dimension
pca20D2 = fit!(machine(@pipeline(Standardizer(), PCA()), train_data_treated))
gr(); 
bpPCA20D2 = biplot(pca20D2) 
png(bpPCA20D2, "PLOTS/Biplot_PCA_20D.png")
#########
## PCA : FINDS THE DIRECTION OF LARGEST VARIANCE 

#PCA, with variance ratio 1
pca = fit!(machine(PCA(variance_ratio = 1), train_data_treated), verbosity = 1);
#biplots
gr(); #gr() modulus 
bp1 = biplot(pca) 
png(bp1, "PLOTS/Biplot_PCA_1.png") 
bp2 = biplot(pca, pc = (1, 3)) 
png(bp2, "PLOTS/Biplot_PCA_2.png")

#PCA ON SMALLER PREDICTORS SETS
pca = fit!(machine(PCA(variance_ratio = 0.9), train_data_treated, verbosity = 1));
report(pca)
fitted_params(pca) # shows the loadings as columns

