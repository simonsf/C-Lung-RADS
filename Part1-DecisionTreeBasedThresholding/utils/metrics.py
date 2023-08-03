import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_auc_score

def iv_categorical_plain(target, feature, feature_name='feature', return_table=False, merge_lv4=True):
    '''
    Calculates the information value of a single categorical feature.  Make sure that `target` is
    a BINARY column where `0` implies non-event and `1` implies event.
    
    Parameters
    ----------
            target : array_like
                Input array or object that can be converted to an array where the dependent variable is stored.
                The target must be binary, where `0` implies non-event and `1` implies event. Must be the same
                length as `feature`.
            feature : array_like
                Input array or object that can be converted to an array where the independent variable is stored.
                Must be the same length as `target`.
            feature_name : str, default 'feature'
                Name of the column (optional). It will default to `feature` when unspecified.
            return_table : boolean, default `False`
                If `return_table == True`, it will return a summary table with the results grouped by bins.
                If `return_table == False`, it will only return a list in the form `[feature, information value]`
            merge_lv4: boolean, default `True`
                if `merge_lv4 == True`, samples with feature(level) 4 will be merged to samples with label 3
    '''
    t = pd.DataFrame({feature_name:feature, 'target':target})
    if merge_lv4 is True:
        t.loc[t[feature_name]==4, feature_name] = 3
    t = t.groupby(feature_name).agg({'target':['size','sum']}).reset_index()
    t.columns = ['cat','count','bads']
    t['goods'] = t['count'] - t['bads']
    t['bads_pct'] = t['bads'].div(t['bads'].sum())
    t.loc[t.bads_pct==0, 'bads_pct']  = 1 / t['bads'].sum()
    t['goods_pct'] = t['goods'].div((t['goods']).sum())
    t.loc[t.goods_pct==0, 'goods_pct']  = 1 / t['goods'].sum()
    t['woe'] = np.log(t['goods_pct'].div(t['bads_pct']))
    t['iv'] = (t['goods_pct'] - t['bads_pct']) * t['woe']
    t['iv'].sum()
    if return_table:
        return t
    else:
        print(t)
        return [feature_name, t['iv'].sum()]
        

def iv_categorical_weighted(target, feature, weight, feature_name='feature', return_table=False, merge_lv4=True):
    '''
    Calculates the information value of a single categorical feature.  Make sure that `target` is
    a BINARY column where `0` implies non-event and `1` implies event.
    
    Parameters
    ----------
            target : array_like
                Input array or object that can be converted to an array where the dependent variable is stored.
                The target must be binary, where `0` implies non-event and `1` implies event. Must be the same
                length as `feature`.
            feature : array_like
                Input array or object that can be converted to an array where the independent variable is stored.
                Must be the same length as `target`.
            weight : array_like
                Input array or object that can be converted to an array where the independent variable is stored.
                Weight of each sample. Must be the same length as `target`.
            feature_name : str, default 'feature'
                Name of the column (optional). It will default to `feature` when unspecified.
            return_table : boolean, default `False`
                If `return_table == True`, it will return a summary table with the results grouped by bins.
                If `return_table == False`, it will only return a list in the form `[feature, information value]`
             merge_lv4: boolean, default `True`
                if `merge_lv4 == True`, samples with feature(level) 4 will be merged to samples with label 3

    '''
    t = pd.DataFrame({feature_name:feature, 'target':target, 'weight':weight})
    if merge_lv4 is True:
        t.loc[t[feature_name]==4, feature_name] = 3
    count = t.groupby([feature_name]).sum()['weight']
    bads = t.groupby([feature_name, 'target']).sum()['weight'].unstack(1)[1]
    t = pd.concat([count,bads],axis=1).reset_index()
    t.columns = ['cat','count','bads']
    t['goods'] = t['count'] - t['bads']
    t['bads_pct'] = t['bads'].div(t['bads'].sum())
    t.loc[t.bads_pct==0, 'bads_pct']  = 1 / t['bads'].sum()
    t['goods_pct'] = t['goods'].div((t['goods']).sum())
    t.loc[t.goods_pct==0, 'goods_pct']  = 1 / t['goods'].sum()
    t['woe'] = np.log(t['goods_pct'].div(t['bads_pct']))
    t['iv'] = (t['goods_pct'] - t['bads_pct']) * t['woe']
    t['iv'].sum()
    if return_table:
        return t
    else:
        print(t)
        return [feature_name, t['iv'].sum()]
        
        
def iv_categorical(target, feature, feature_name='feature', return_table=False, weight=None, merge_lv4=True):
    if weight is None:
        return iv_categorical_plain(target, feature, feature_name, return_table, merge_lv4=merge_lv4)
    else:
        return iv_categorical_weighted(target, feature, weight=weight, 
                                       feature_name=feature_name, return_table=return_table, merge_lv4=merge_lv4)