function data_split(data;
    shuffle = false,
    idx_train = 1:50,
    idx_valid = 51:100,
    idx_test = 101:500)
idxs = if shuffle
randperm(size(data, 1))
else
1:size(data, 1)
end
(train = data[idxs[idx_train], :],
valid = data[idxs[idx_valid], :],
test = data[idxs[idx_test], :])
end
function fit_and_evaluate(model, data)
mach = fit!(machine(model, select(data.train, :x), data.train.y),
verbosity = 0)
(train = rmse(predict(mach, select(data.train, :x)), data.train.y),
valid = rmse(predict(mach, select(data.valid, :x)), data.valid.y),
test = rmse(predict(mach, select(data.test, :x)), data.test.y))
end

function cross_validation_sets(idx, K)
    n = length(idx)
    r = n ÷ K
    [let idx_valid = idx[(i-1)*r+1:(i == K ? n : i*r)]
         (idx_valid = idx_valid, idx_train = setdiff(idx, idx_valid))
     end
     for i in 1:K]
end
function cross_validation(model, data; K = 5, shuffle = false)
    idxs = 1:size(data, 1)
    if shuffle
        idxs = Random.shuffle(idxs)
    end
    losses = [fit_and_evaluate(model,
                               data_split(data;
                                          idx_test = [], # no test set
                                          idxs...)) # training and validation
              for idxs in cross_validation_sets(idxs, K)]
    (train = mean(getproperty.(losses, :train)),
     valid = mean(getproperty.(losses, :valid)))
end
