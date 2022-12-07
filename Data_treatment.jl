using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, CSV ,Statistics

train_data = CSV.read("DATA/train.csv", DataFrame)
coerce!(train_data, :labels => Multiclass)
#dropmissing was tested and output is of same size -> no missing Data
dropmissing!(train_data);
X = select(train_data, Not(:labels))
y = train_data.labels
#removal of constant parameters:
X_const = X[:, std.(eachcol(X)) .!= 0]

#removal of perfectly correlated redundant parameters:
findall(≈(1), cor(Matrix(X_const))) |> # find all indices with correlation ≈ 1
idxs -> filter(x -> x[1] > x[2], idxs)
#corr = findall(≈(1), cor(Matrix(X_const))) |> idxs -> filter(x -> x[1] > x[2], idxs)
#a = 1:size(corr)[1]
#corr_colnames = [names(X_const)[i] for i in a]
#X_cleaned = X_const
#for i in a
    #X_cleaned = select(X_cleaned, Not([i]))
#end

#dataX = X_const
#datay = y

#test_data = CSV.read("DATA/test.csv", DataFrame)
CSV.write("DATA/dataX.csv", X_const)


#
