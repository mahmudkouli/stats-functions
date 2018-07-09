def vif(model, X, target_name):
	
    columns_without_j = list(X.columns)
    columns_without_j.remove(target_name)

    model.fit(X[columns_without_j], X[target_name])
    R_sq = model.score(X[columns_without_j], X[target_name])
    
    columns_without_j.append(target_name)

    return 1/(1-R_sq)