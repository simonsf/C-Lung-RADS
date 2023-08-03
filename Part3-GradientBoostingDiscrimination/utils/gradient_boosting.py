import copy
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV

#basic loss and gradient tools
def huber_prob(p, min=0.3, max=0.7):
    if p<=min:
        return min
    if p >= max:
        return max
    return p
    
def prob_thres(p, thres=[0.4, 0.7]):
    oup = 0.1
    for t in thres:
        if p > t:
            oup = t
    return oup

def SquaredLoss_NegGradient(y_pred, y):
    return y - y_pred

def Huberloss_NegGradient(y_pred, y, alpha=0.3):
    diff = y - y_pred
    delta = stats.scoreatpercentile(np.abs(diff), alpha * 100)
    g = np.where(np.abs(diff) > delta, delta * np.sign(diff), diff)
    return g

def logistic(p):
    return 1 / (1 + np.exp(-2 * p))
    
def logistic_inv(p):
    return (np.log(p) - np.log(1-p)) / 2

def LogisticLoss_NegGradient(y_pred, y):
    g = 2 * y / (1 + np.exp(1 + 2 * y * y_pred))  # logistic_loss = log(1+exp(-2*y*y_pred))
    return g

def modified_huber(p):
    return (np.clip(p, -1, 1) + 1) / 2
    
def modified_huber_inv(p):
    return np.clip(p * 2 - 1, -1, 1)

def Modified_Huber_NegGradient(y_pred, y):
    margin = y * y_pred
    g = np.where(margin >= 1, 0, np.where(margin >= -1, y * 2 * (1-margin), 4 * y))
    # modified_huber_loss = np.where(margin >= -1, max(0, (1-margin)^2), -4 * margin)
    return g


def cal_gradient(y_pred, y, method, loss_type=None):
    if method == 'regression':
        if loss_type is None:
            loss_type = 'square'
        if loss_type == 'huber':
            grad = Huberloss_NegGradient(y_pred, y)
        elif loss_type == 'square':
            grad = SquaredLoss_NegGradient(y_pred, y)
        else:
            raise ValueError('Unsupported loss type')
    elif method == 'classification':
        if loss_type is None:
            loss_type = 'logistic'
        if loss_type == "modified_huber":
            grad = Modified_Huber_NegGradient(modified_huber_inv(y_pred), y)
        elif loss_type == 'logistic':
            grad = LogisticLoss_NegGradient(logistic_inv(y_pred), y)
        else:
            raise ValueError('Unsupported loss type')
    return grad


# subsample method for decorrelate Sex and Smoking
def sex_balance(df, sex_col='Sex', minor_frac=0.8):
    mean = df[sex_col].mean()
    minor = 1 if mean < 0.5 else 0
    df_minor = df[df[sex_col]==minor]
    df_major = df[df[sex_col]==(1-minor)]
    
    df_minor_frac = df_minor.sample(frac=minor_frac)
    major_frac = len(df_minor_frac) / len(df_major)
    df_major_frac = df_major.sample(frac=major_frac)
    return pd.concat([df_minor_frac, df_major_frac])
    
    
def sex_can_balance_subsample(df, tag_col='Cancer', sex_col='Sex', minor_frac=0.8):
    df_can = df[df[tag_col]==1]
    df_bn = df[df[tag_col]==0]
    
    df_can_frac = sex_balance(df_can, sex_col, minor_frac)
    df_bn_frac = sex_balance(df_bn, sex_col, minor_frac)
    
    return pd.concat([df_bn_frac, df_can_frac]).sample(frac=1)


# model fitting tools
def cal_sample_weight(label, weight_pos=None):
    if weight_pos is not None:
        if weight_pos == 'balanced':
            num_pos = np.sum(label==1)
            num_neg = len(label)-num_pos
            weight_pos = float(num_neg) / float(num_pos)
        else:
            weight_pos = float(weight_pos)
    else:
        weight_pos = 1
    class_weight = [1 if l==-1 else weight_pos for l in label]
    return class_weight
    
def fitting(x, y, label, weight_pos=None, intercept=True):
    clf = define_lasso(intercept=intercept)

    class_weight = cal_sample_weight(label, weight_pos)
    
    clf.fit(x, y, sample_weight=class_weight)
    #clf.fit(x, y)
    return clf
    
def define_lasso(intercept=True):
    return LassoCV(eps=3e-4, tol=1e-5,cv=5, max_iter=5000, n_alphas=1000, fit_intercept=intercept, random_state=1)


#

def f(x):
    return x

def boosting_onestep_coef(df0, 
                        image_col, basic_cols, tag_col, descrete_cols,
                        method='classification', 
                        loss_type=None, image_func=f, 
                        fillna=True, fillval=0, 
                        weight_pos=None, 
                        dummy=True, use_intercept=True):
    
    df = copy.deepcopy(df0)

    #training
    if fillna is True:
        df[basic_cols] = df[basic_cols].fillna(fillval)
        df[image_col] = df[image_col].fillna(0.01)
    else:
        df = df[~df[basic_cols].isna().transpose().any()]
    
    x = df[basic_cols]
    
    df[image_col] = [image_func(i) for i in df[image_col]]
    if dummy is True:
        cols = [c for c in basic_cols if c in descrete_cols]
        for c in cols:
            x[c] = x[c].astype(np.int)
            if len(x[c].drop_duplicates()) == 1:
                x.loc[0, c] = 1
        x = pd.get_dummies(x, columns=cols)
        
    label = np.array(df[tag_col])
    label[label<=0] = -1
    y_pred = np.array(df[image_col]) #> 0.5
    #y_pred[y_pred<=0] = -1
    y = cal_gradient(y_pred, label, method, loss_type)
    #y[np.abs(y)<=0.2] = y[np.abs(y)<=0.2] / 2
    
    lasso = fitting(x, y, label, weight_pos=weight_pos, intercept=use_intercept)
    
    return lasso


def bag_lasso(lasso_list):
    coef_list= [lasso.coef_.reshape(-1,1) for lasso in lasso_list]
    inter_list = [lasso.intercept_ for lasso in lasso_list]
    coef = np.around(np.mean(coef_list, axis=0), 2)
    intercept = np.mean(inter_list)
    return coef, intercept 

def regression_from_coef(x, coef, intercept):
    predict = np.matmul(np.array(x), coef).reshape(1,-1)[0] + intercept
    return predict

def boost_lasso_predict(df_test, image_col, basic_cols, descrete_cols, coef, intercept, feature_names=None, fillna=True, fillval=0, dummy=True,image_func=f,loss_type=None,lr=1,pred_name='pred'):
    df1 = copy.deepcopy(df_test)
    if fillna is True:
        df1[basic_cols] = df1[basic_cols].fillna(fillval)
    else:
        df1 = df1[~df1[basic_cols].isna().transpose().any()]
    #detesting
    x1 = df1[basic_cols]
    df1[image_col] = [image_func(i) for i in df1[image_col]]
    if dummy is True:
        cols = [c for c in basic_cols if c in descrete_cols]
        for c in cols:
            x1[c] = x1[c].astype(np.int)
            if len(x1[c].drop_duplicates()) == 1:
                x1.loc[0, c] = 1
        x1 = pd.get_dummies(x1, columns=cols)
    if feature_names is not None:
        x1 = x1[feature_names]
        
        
    y_pred = regression_from_coef(x1, coef, intercept)
    if loss_type == 'modified_huber':
        y0 = modified_huber_inv(df1[image_col])
        y_pred_total = modified_huber(lr * y_pred + y0)
    else:
        y0 = logistic_inv(df1[image_col])
        y_pred_total = logistic(lr * y_pred + y0)
    #y_true = df1[tag_col]
    df1[pred_name] = y_pred_total
    return df1, x1, y0
    


def boosting_onestep_coef_bagging(df0, sex_col,
                                    image_col, basic_cols, tag_col, descrete_cols,
                                    lr=1, method='classification', 
                                    N=50, minor_frac=0.8,
                                    loss_type=None, image_func=f, 
                                    fillna=True, fillval=0, 
                                    weight_pos=None, 
                                    dummy=True, use_intercept=True):
    
    lasso_list = []
    for _ in range(N):
        df = sex_can_balance_subsample(df0, tag_col, sex_col=sex_col, minor_frac=minor_frac)
        lasso = boosting_onestep_coef(df, image_col, basic_cols, tag_col, descrete_cols, 
                                      method, loss_type, image_func, 
                                      fillna, fillval, 
                                      weight_pos, 
                                      dummy, use_intercept)
        lasso_list.append(lasso)

    coef, intercept = bag_lasso(lasso_list)
    feature_names = list(lasso_list[0].feature_names_in_)
    df_pred, _, _ = boost_lasso_predict(df0, image_col, basic_cols, descrete_cols, coef, intercept, feature_names, fillna, 
                                  fillval, dummy, image_func, loss_type, lr, pred_name='pred_cli0')
    lasso = boosting_onestep_coef(df_pred, 'pred_cli0', [sex_col], tag_col, [sex_col],
                                    method, loss_type, f, 
                                    fillna, fillval, 
                                    weight_pos, 
                                    dummy, use_intercept)
    coef0 = lasso.coef_.reshape(-1,1)
    df_pred, _, _ = boost_lasso_predict(df_pred, 'pred_cli0', [sex_col], [sex_col], coef0, lasso.intercept_, lasso.feature_names_in_,
                                        fillna, fillval, dummy, image_func, loss_type, lr, pred_name='pred_cli')

    feature_names.extend(list(lasso.feature_names_in_))
    coef = np.concatenate((coef,coef0))
    intercept = intercept + lasso.intercept_

    return df_pred, feature_names, coef, intercept





    

