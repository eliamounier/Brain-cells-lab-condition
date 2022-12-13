using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, OpenML, DataFrames, CSV, MLCourse, Random, Statistics, MLJ, MLJLinearModels, Distributions

train_data = CSV.read("DATA/train.csv", DataFrame)
test_data = CSV.read("DATA/test.csv", DataFrame)
#dropmissing was tested and output is of same size -> no missing Data
X = select(train_data, Not(:labels))
X_totreat = vcat(X, test_data)
datay = categorical(train_data.labels, levels = ["KAT5", "CBP", "eGFP"], ordered = true)
X_const = X_totreat[:, std.(eachcol(X)) .!= 0]

corr = findall(≈(1), cor(Matrix(X_const))) |> idxs -> filter(x -> x[1] > x[2], idxs)
a = 1:size(corr)[1]
corr_indexes1 = [corr[i][1] for i in a]
corr_names1 = [names(X_const)[j] for j in corr_indexes1]
corr_names1_wout = unique(corr_names1)
X_cleaned = select(X_const, Not(corr_names1_wout))


dataX = X_cleaned[1:5000, :]
dataT = X_cleaned[5001:8093, :]


CSV.write("DATA/datayy.csv", datay, writeheader = true)
CSV.write("DATA/dataX.csv", dataX)
CSV.write("DATA/dataT.csv", dataT)