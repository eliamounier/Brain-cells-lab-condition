using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, CSV, Statistics, MLJ, Serialization

#importing datasets
train_data = CSV.read("DATA/train.csv", DataFrame)
test_data = CSV.read("DATA/test.csv", DataFrame)
train_output = categorical(train_data.labels, levels = ["KAT5", "eGFP", "CBP"], ordered = true)

#dropmissing was tested and output is of same size -> no missing Data

#REMOVAL OF CONSTANT PREDICTORS
X = select(train_data, Not(:labels))
X_totreat = vcat(X, test_data)
y = train_data.labels
X_const = X_totreat[:, std.(eachcol(X)) .!= 0]

#REMOVAL OF COORELATED PREDICTORS
corr = findall(≈(1), cor(Matrix(X_const))) |> idxs -> filter(x -> x[1] > x[2], idxs)
a = 1:size(corr)[1]
corr_indexes1 = [corr[i][1] for i in a]
corr_names1 = [names(X_const)[j] for j in corr_indexes1]
corr_names1_wout = unique(corr_names1)
X_cleaned = select(X_const, Not(corr_names1_wout))

#back into exploitable form
dataX = X_cleaned[1:5000, :]
dataT = X_cleaned[5001:8093, :]

#Saving new datasets
serialize("DATA/trainlabels.dat", train_output)
CSV.write("DATA/trainX.csv", dataX)
CSV.write("DATA/testX.csv", dataT)
