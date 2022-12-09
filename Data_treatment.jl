using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, OpenML, DataFrames, CSV, MLCourse, Random, Statistics, MLJ, MLJLinearModels, Distributions

train_data = CSV.read("DATA/train.csv", DataFrame)
test_data = CSV.read("DATA/test.csv", DataFrame)
coerce!(train_data, :labels => Multiclass)
#dropmissing was tested and output is of same size -> no missing Data
X = select(train_data, Not(:labels))
X_totreat = vcat(X, test_data)
y = train_data.labels
X_const = X_totreat[:, std.(eachcol(X)) .!= 0]

corr = findall(≈(1), cor(Matrix(X_const))) |> idxs -> filter(x -> x[1] > x[2], idxs)
a = 1:size(corr)[1]
corr_indexes1 = [corr[i][1] for i in a]
corr_names1 = [names(X_const)[j] for j in corr_indexes1]
# X_cleaned = select(X_const, Not(corr_names1))


dataX = X_const[1:5000, :]
dataT = X_const[5001:8093, :]
datay = y


#CSV.write("DATA/datay.csv", y)
#CSV.write("DATA/dataX.csv", X_const)