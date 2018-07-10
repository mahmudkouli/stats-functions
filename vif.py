def vif(model, X, target_name):
    """Return VIF score of a variable.
    
    Function returns VIF score of a specified variable by using the auxiliarry regression method approach to 
    Multiple Linear Regression. It is built using the Scikit Learn Linear Regression class.
    
    Parameters
    ----------    
    model: class instance of Scikit Learn LinearRegression
    X: dataframe of feature variables
    target_name: variable for which VIF score will be calculated for as a string
    
    Returns
    -------
    float
        Value for VIF score of the target_name feature
    """
    columns_without_j = list(X.columns)
    columns_without_j.remove(target_name)
    model.fit(X[columns_without_j], X[target_name])
    R_sq = model.score(X[columns_without_j], X[target_name])
    
    columns_without_j.append(target_name)
    return 1/(1-R_sq)