# ML_chloelia

The aim of this project is to classify cells under 3 different studying conditions according to their genes expressions.

## Installation of the environment
Same environment as the BIO-322 course
1) Julia version 1.7.3 needs to be pre-installed
2) MLcourse installation:
launch julia and run the following code to install the course material:

```julia
julia> using Pkg
Pkg.activate(temp = true)
Pkg.develop(url = "https://github.com/jbrea/MLCourse")
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
Pkg.instantiate()
using MLCourse
MLCourse.start()
```

## After first Git Pull
Data sets "test.csv" and "train.csv" need to be downloaded from: 

- https://lcnwww.epfl.ch/bio322/project2022/test.csv.gz 
- https://lcnwww.epfl.ch/bio322/project2022/train.csv.gz  

and added into the empty folder "DATA" under the following paths: "DATA/test.csv" and "DATA/train.csv".


## Running the code
Before runing the different models, first Data_treatment.jl and then PCA.jl need to be executed. The new treated data will be stored in the DATA folder.

It is afterwards possible to run all the remaining files. Here is a brief summary of the content of each file : 
- Regularization.jl : LogisticClassifier, LogisticClassifier with a lasso regularization, an attempt at LogisticClassifier with a ridge regularization.
- KNNandPN.jl : KNNClassifier and polynomial classifier models. 
- non_linear_methods.jl :
    - models with cleaned data before PCA: Random forest, various tests of Neural networks first without and then with regularization.
    - models with cleaned data treated by PCA: Gradient Boosting Trees, Neuron netowrks (preliminary tests with and without regularization, tuning of epochs and dropout).
- Clusters.jl : KMeans clusering and confusion matrices.
- FinalModels.jl : Contains the 2 models we believe are the best : 
    - LogisticClassifier with PCA treated data and lasso regularization (also in Regularization.jl).
    - Neural network with PCA treated data, lasso regularization and dropout (also in non_linear_methods.jl). 
- Visualisation.ipynb : gives a look at raw data, cleaned data, and PCA. 

The results on test set and plots will be stored respectively in the RESULTS and the PLOTS folders. 

### Report and Kaggle final submissions
- Our report can be found under report.pdf.
- To reproduce the models from the kaggle submission, start by running first Data_treatment.jl and then PCA.jl. The final models are all in our original files but have as well been copied to FinalModel.jl and can be ran from there.

