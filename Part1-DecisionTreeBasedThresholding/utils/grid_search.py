import copy
import pandas as pd
from itertools import product
from .metrics import *


##### Variable split methods

def split_var_i(num, thres, make_int=False):
    if make_int is True:
        num = round(num)
    for i in range(len(thres)):
        if num < thres[i]:
            return i + 1
    return i + 2
    

def split_var(data, var, thres, new_var):
    ''' 
    Variable discretization 

    Parameters
    ----------
            data: pandas DataFrame, dataset to be operated
            var: Str, name of variable to be discretized
            thres: List, thresholds for each discrete level
            new_var: Str, name of discretized varibale
    Returns:
            data: pandas DataFrame, dataset operated, with new column of discretized varibale
    ----------
            
    '''    
    var_discrete = [split_var_i(v, thres) for v in data[var]]
    data[new_var] = var_discrete
    return data


def split_mix_i(x, x_solid, axis_thres, solid_thres, make_int=False):
   if make_int is True:
       x = round(x)
       x_solid = round(x_solid)
   solid_thres = sorted(solid_thres)
   axis_thres = sorted(axis_thres)
   if x < axis_thres[0] and x_solid < solid_thres[0]:
       return 1
   if x < axis_thres[1] and x_solid < solid_thres[0]:
       return 2
   if x < axis_thres[1] and x_solid < solid_thres[1]:
       return 3
   return 4
   
def split_mix(data, var, thres, new_var):
    ''' 
    Variable discretization, especially designed for mGGN data

    Parameters
    ----------
            data: pandas DataFrame, dataset to be operated
            var: List of Str, name of variables to be discretized. 
            thres: dict, thresholds for each discrete level
            new_var: Str, name of discretized varibale
    Returns:
            data: pandas DataFrame, dataset operated, with new column of discretized varibale
    ----------
    '''    
    assert len(var)==2
    var_discrete = [split_mix_i(data[var[0]].iloc[i], data[var[1]].iloc[i], thres['axis'], thres['solid']) for i in range(len(data[var]))]
    data[new_var] = var_discrete
    return data


def split_all_data(data, var, solid_thres, glass_thres, mix_thres, new_var):
    ''' 
    Nodule thresholding based on given thresholds of each density type

    Parameters
    ----------
            data: pandas DataFrame, dataset to be operated
            var: List of Str, name of variables to be discretized. 
                 First item is the name of total axis, second item the name of solid axis
            solid_thres: List, axis thresholds for each level of solid nodule
            glass_thres: List, axis thresholds for each level of pGGN nodule
            mix_thres: Dict, axis and solid part axis thresholds for each level of mGGN nodules
                       should have key 'axis' and 'solid' each for total axis and solid part axis
            new_var: Str, name of thresholding output
    Returns:
            data: pandas DataFrame, dataset operated, with new column of discretized varibale
    ----------
    '''    

    assert len(var)==2
    solid = copy.deepcopy(data[data.NoduleType==1])
    glass = copy.deepcopy(data[data.NoduleType==2])
    mix = copy.deepcopy(data[data.NoduleType==3])
    
    solid = split_var(solid, var[0], solid_thres, new_var)
    glass = split_var(glass, var[0], glass_thres, new_var)
    mix = split_mix(mix, var, mix_thres, new_var)
    
    data_ = pd.concat([solid, glass, mix])
    return data_

def lungrads_4x(data, var, mal='MalignantProb', rename=None, thres=0.5):
    ''' 
    Give nodule with high malignant probablity different level

    Parameters
    ----------
            data: pandas DataFrame, dataset to be operated
            var: Str, name of discretized varibale
            mal: Str, name of nodule malignant probablity column
            rename: None or numeric, new level of nodule with high malignancy
            thres: float, threshold of malignant probablity to be considered high malignancy
    Returns:
            data: pandas DataFrame, dataset operated
    ----------
    '''   
    if rename is None:
        rename = 4
    
    data.loc[(data[mal]>thres)&(data[var].isin([2,3])), var] = rename
    return data


### Threshold evaluating methods
def judge_single_thres(data0, var, target, thres, malig=0.5, malig_col='MalignantProb', weight=None):
    ''' 
    Calculate metrics (AUC, IV) of a given discretization threshold

    Parameters
    ----------
            data0: pandas DataFrame, dataset to be operated
            var: Str, name of variable to be discretized
            target: Str, name of binary target label
            thres: List, thresholds for each discrete level
            malig: None or float, threshold of malignant probablity to be considered high malignancy
            malig_col: None or str, name of nodule malignant probablity column
                       Should be provided if malig is not None
            weight: None or str, name of sample weight column in dataset
    Returns:
            results: Dict
    ----------
    '''   
    data = copy.deepcopy(data0)
    data = split_var(data, var, thres, 'Axis_int')
    
    if weight is None:
        sample_weight=None
    else:
        sample_weight = data[weight]
        
    t0 = iv_categorical(data[target], data['Axis_int'], return_table=True, weight=sample_weight)
    auc = roc_auc_score(data[target], data['Axis_int'], sample_weight=sample_weight)
    results = {'woe':np.array(t0.woe), 'iv':np.array(t0.iv), 'iv_sum':t0.iv.sum(), 'auc':auc}
    
    if malig is not None:
        data = lungrads_4x(data, 'Axis_int',mal=malig_col, thres=malig)
        t1 = iv_categorical(data[target], data['Axis_int'], return_table=True, weight=sample_weight)
        auc = roc_auc_score(data[target], data['Axis_int'], sample_weight=sample_weight)
        results['woe_4x'] = np.array(t1.woe)
        results['iv_4x'] = np.array(t1.iv)
        results['iv_sum_4x'] = t1.iv.sum()
        results['auc_4x'] = auc
    for key in results.keys():
        results[key] = np.around(results[key], 2)
    return results
    
def judge_single_thres_mix(data0, var, target, thres, malig=0.5, malig_col='MalignantProb', weight=None):
    ''' 
    Calculate metrics (AUC, IV) of a given discretization threshold, especially designed for mGGN nodules.

    Parameters
    ----------
            data0: pandas DataFrame, dataset to be operated
            var: List of Str, name of variables to be discretized. 
                 First item is the name of total axis, second item the name of solid axis
            target: Str, name of binary target label
            thres: Dict, axis and solid part axis thresholds for each level of mGGN nodules
                   should have key 'axis' and 'solid' each for total axis and solid part axis
            malig: None or float, threshold of malignant probablity to be considered high malignancy
            malig_col: None or str, name of nodule malignant probablity column
                       Should be provided if malig is not None
            weight: None or str, name of sample weight column in dataset
    Returns:
            results: Dict
    ----------
    '''   
    data = copy.deepcopy(data0)
    data = split_mix(data, var, thres, 'Axis_int')
    
    if weight is None:
        sample_weight=None
    else:
        sample_weight = data[weight]
        
    t0 = iv_categorical(data[target], data['Axis_int'], return_table=True, weight=sample_weight, merge_lv4=False)
    auc = roc_auc_score(data[target], data['Axis_int'], sample_weight=sample_weight)
    results = {'woe':np.array(t0.woe), 'iv':np.array(t0.iv), 'iv_sum':t0.iv.sum(), 'auc':auc}
    
    if malig is not None:
        data = lungrads_4x(data, 'Axis_int', mal=malig_col, thres=malig)
        t1 = iv_categorical(data[target], data['Axis_int'], return_table=True, weight=sample_weight, merge_lv4=False)
        auc = roc_auc_score(data[target], data['Axis_int'], sample_weight=sample_weight)
        results['woe_4x'] = np.around(np.array(t1.woe), 2)
        results['iv_4x'] = np.around(np.array(t1.iv), 2)
        results['iv_sum_4x'] = np.around(t1.iv.sum(), 2)
        results['auc_4x'] = np.around(auc, 2)
    for key in results.keys():
        results[key] = np.around(results[key], 2)
    return results
    
def judge_single_thres_alldens(data0, var, target, 
                               solid_thres, glass_thres, mix_thres, 
                               malig=None, malig_col='MalignantProb', 
                               split=True, weight=None):
    ''' 
    Calculate metrics (AUC, IV) of a given discretization threshold, designed for all three types of nodules.

    Parameters
    ----------
            data0: pandas DataFrame, dataset to be operated
            var: List of Str, name of variables to be discretized. 
                 First item is the name of total axis, second item the name of solid axis
            target: Str, name of binary target label
            solid_thres: List, axis thresholds for each level of solid nodule
            glass_thres: List, axis thresholds for each level of pGGN nodule
            mix_thres: Dict, axis and solid part axis thresholds for each level of mGGN nodules
                       should have key 'axis' and 'solid' each for total axis and solid part axis
            malig: None or float, threshold of malignant probablity to be considered high malignancy
            malig_col: None or str, name of nodule malignant probablity column
                       Should be provided if malig is not None
            split: boolean, default = `True`. Whether to discretize all the nodules.
                   Set to `False` only when data is pre-discretized and contains column `Axis_int`
            weight: None or str, name of sample weight column in dataset
    Returns:
            results: Dict
    ----------
    '''   
    
    if split is True:
        data = split_all_data(data0, var, solid_thres, glass_thres, mix_thres, 'Axis_int')
    else:
        data = data0
    if weight is None:
        sample_weight=None
    else:
        sample_weight = data[weight]
    t0 = iv_categorical(data[target], data['Axis_int'], return_table=True, weight=sample_weight)
    t0_l4 = iv_categorical(data[target], data['Axis_int'], return_table=True, weight=sample_weight, merge_lv4=False)
    auc = roc_auc_score(data[target], data['Axis_int'], sample_weight=sample_weight)
    results = {'woe':np.array(t0.woe), 'iv':np.array(t0.iv), 'iv_sum':t0.iv.sum(), 'auc':auc, 'woe_l4':np.array(t0_l4.woe), 'iv_l4':np.array(t0_l4.iv), 'iv_sum_l4':t0_l4.iv.sum()}
    if malig is not None:
        data = lungrads_4x(data, 'Axis_int', mal=malig_col, thres=malig)
        t1 = iv_categorical(data[target], data['Axis_int'], return_table=True, weight=sample_weight, merge_lv4=False)
        auc = roc_auc_score(data[target], data['Axis_int'], sample_weight=sample_weight)
        results['woe_4x'] = np.array(t1.woe)
        results['iv_4x'] = np.array(t1.iv)
        results['iv_sum_4x'] = t1.iv.sum()
        results['auc_4x'] = auc
    for key in results.keys():
        results[key] = np.around(results[key], 2)
    return results


## grid search methods
def grid_search_solid(data, 
                      solid_range, 
                      var, target, 
                      malig=None, malig_col='MalignantProb', 
                      sort_key='iv_sum', 
                      min_rate=[0.6,0.05,0.01,0.005], 
                      weight=None):
    ''' 
    Search all threshold combinations for solid nodules, calculate metrics and sort

    Parameters
    ----------
            data0: pandas DataFrame, dataset to be operated
            solid_range: Dict, candidates of axis thresholds in each level
                         Template:
                        solid_range = {
                                        1:[4,5,6,7],
                                        2:[8,9,10,11],
                                        3:[14,15,16,17]
                                      }
            var: Str, name of variable to be discretized
            target: Str, name of binary target label
            malig: None or float, threshold of malignant probablity to be considered high malignancy
            malig_col: None or str, name of nodule malignant probablity column
                       Should be provided if malig is not None
            sort_key: Str, name of metric to be sort on.
                      Should be one of ['iv_sum', 'iv_sum_4x', 'auc', 'auc_4x']
            malig: None or float, threshold of malignant probablity to be considered high malignancy
            malig_col: None or str, name of nodule malignant probablity column
                       Should be provided if malig is not None
            min_rate: array_like, min proportion of each level in all population.
                      Length should be 4, correspondence of level 1 to 4
            weight: None or str, name of sample weight column in dataset
    Returns:
            results: DataFrame, dataset of each threshold combination, sorted by certain metric
    ----------
    '''   
    num = len(data)
    results = []
    for a1 in solid_range[1]:
        n1 = len(data[data[var]<a1])
        
        if n1 < num * min_rate[0]:
            continue
        for a2 in solid_range[2]:
            n2 = len(data[data[var]<a2]) - n1
            if n2 < num * min_rate[1]:
                continue
            for a3 in solid_range[3]:
                n3 = len(data[data[var]<a3]) - n1 - n2
                if n3 < num * min_rate[2]:
                    continue
                n4 = len(data[data[var]>=a3])
                if n4 < num * min_rate[3]:
                    continue
                solid_thres = [a1, a2, a3]
                result = judge_single_thres(data, var, target, solid_thres, malig=malig, malig_col=malig_col,weight=weight)
                if list(result['woe']) != sorted(list(result['woe']), reverse=True):
                        continue
                for key in result:
                    if isinstance(result[key], np.ndarray):
                        result[key] = np.around(result[key], decimals=2)
                result['thres'] = solid_thres
                results.append(result)
    return pd.DataFrame(results).sort_values(sort_key, ascending=False)


def grid_search_glass(data, 
                      glass_range, 
                      var, target, 
                      malig=None, malig_col='MalignantProb', 
                      sort_key='iv_sum', 
                      min_rate=[0.5, 0.1, 0.05], 
                      weight=None):
    ''' 
    Search all threshold combinations for pGGN, calculate metrics and sort

    Parameters
    ----------
            data0: pandas DataFrame, dataset to be operated
            glass_range: Dict, candidates of axis thresholds in each level
                         Template:
                         glass_range = {
                                            1:[5,6,7,8,9,10],
                                            2:[14,15,16,17],
                                        }
            var: Str, name of variable to be discretized
            target: Str, name of binary target label
            malig: None or float, threshold of malignant probablity to be considered high malignancy
            malig_col: None or str, name of nodule malignant probablity column
                       Should be provided if malig is not None
            sort_key: Str, name of metric to be sort on.
                      Should be one of ['iv_sum', 'iv_sum_4x', 'auc', 'auc_4x']
            malig: None or float, threshold of malignant probablity to be considered high malignancy
            malig_col: None or str, name of nodule malignant probablity column
                       Should be provided if malig is not None
            min_rate: array_like, min proportion of each level in all population.
                      Length should be 3, correspondence of level 1 to 3
            weight: None or str, name of sample weight column in dataset
    Returns:
            results: DataFrame, dataset of each threshold combination, sorted by certain metric
    ----------
    '''   
    
    num = len(data)
    results = []
    for a1 in glass_range[1]:
        n1 = len(data[data[var]<a1])
        
        if n1 < num * min_rate[0]:
            continue
        for a2 in glass_range[2]:
            n2 = len(data[data[var]<a2]) - n1
            if n2 < num * min_rate[1]:
                continue
            n3 = len(data[data[var]>=a2])
            if n3 < num * min_rate[2]:
                continue
            glass_thres = [a1, a2]
            result = judge_single_thres(data, var, target, glass_thres, malig=malig, malig_col=malig_col, weight=weight)
            if list(result['woe']) != sorted(list(result['woe']), reverse=True):
                        continue
            for key in result:
                if isinstance(result[key], np.ndarray):
                    result[key] = np.around(result[key], decimals=2)
            result['thres'] = glass_thres
            results.append(result)
    return pd.DataFrame(results).sort_values(sort_key, ascending=False)


def grid_search_mix(data, 
                    total_range,
                    solid_range, 
                    var, target, 
                    malig=None, malig_col='MalignantProb', 
                    sort_key='iv_sum', 
                    min_rate=[0.6,0.05,0.01,0.005], 
                    weight=None):
    ''' 
    Search all threshold combinations for mGGN, calculate metrics and sort

    Parameters
    ----------
            data0: pandas DataFrame, dataset to be operated
            total_range: Dict, candidates of total axis thresholds in each level
                         Template:
                         total_range = {
                                            1:[5,6,7,8],
                                            2:[10,11,12,13,14,15],
                                       }
            solid_range: Dict, candidates of solid part axis thresholds in each level
                         Template:
                         solid_range = {
                                            1:[5,6,7,8],
                                            2:[9,10,11,12,13],
                                       }
            var: List of Str, name of variables to be discretized. 
                 First item is the name of total axis, second item the name of solid axis
            target: Str, name of binary target label
            malig: None or float, threshold of malignant probablity to be considered high malignancy
            malig_col: None or str, name of nodule malignant probablity column
                       Should be provided if malig is not None
            sort_key: Str, name of metric to be sort on.
                      Should be one of ['iv_sum', 'iv_sum_4x', 'auc', 'auc_4x']
            malig: None or float, threshold of malignant probablity to be considered high malignancy
            malig_col: None or str, name of nodule malignant probablity column
                       Should be provided if malig is not None
            min_rate: array_like, min proportion of each level in all population.
                      Length should be 4, correspondence of level 1 to 4
            weight: None or str, name of sample weight column in dataset
    Returns:
            results: DataFrame, dataset of each threshold combination, sorted by certain metric
    ----------
    '''   

    num = len(data)
    results = []
    v, vs = var
    for a1 in total_range[1]:
        for s1 in solid_range[1]:
            n1 = len(data[(data[v]<a1)&(data[vs]<s1)])
            if n1 < num * min_rate[0]:
                continue
            for a2 in total_range[2]:
                n2 = len(data[(data[v]<a2)&(data[vs]<s1)]) - n1
                if n2 < num * min_rate[1]:
                    continue
                for s2 in solid_range[2]:
                    n3 = len(data[(data[v]<a2)&(data[vs]<s2)]) - n1 - n2
                    if n3 < num * min_rate[2]:
                        continue
                    n4 = num - n1 - n2 - n3
                    if n4 < num * min_rate[3]:
                        continue
                    mix_thres = {'axis':[a1, a2], 'solid':[s1, s2]}
                    result = judge_single_thres_mix(data, var, target, mix_thres, malig=malig, malig_col=malig_col, weight=weight)
                    if list(result['woe']) != sorted(list(result['woe']), reverse=True):
                        continue
                    for key in result:
                        if isinstance(result[key], np.ndarray):
                            result[key] = np.around(result[key], decimals=2)
                    result['thres'] = mix_thres
                    results.append(result)
    return pd.DataFrame(results).sort_values(sort_key, ascending=False)


def grid_search_all(data0, 
                    var, target, 
                    solid_candidates, glass_candidates, mix_candidates, 
                    malig_candidates, malig_col='MalignantProb', 
                    sort_key='iv_sum_4x',
                    min_rate=[0.75,0.05,0.01,0.005], weight=None):
    
    ''' 
    Search all threshold combinations for all density types, calculate metrics and sort

    Parameters
    ----------
            data0: pandas DataFrame, dataset to be operated
            var: List of Str, name of variables to be discretized. 
                 First item is the name of total axis, second item the name of solid axis
            target: Str, name of binary target label
            solid_candidates: List,
                              each item is a combination of solid nodule thresholds
            glass_candidates: List,
                              each item is a combination of pGGN thresholds
            mix_candidates: List,
                            each item is a combination of mGGN thresholds
            malig_candidates: None or array-like, thresholds of malignant probablity to be considered high malignancy
            malig_col: None or str, name of nodule malignant probablity column
                       Should be provided if malig is not None
            sort_key: Str, name of metric to be sort on.
                      Should be one of ['iv_sum', 'iv_sum_4x', 'auc', 'auc_4x']
            malig_col: None or str, name of nodule malignant probablity column
                       Should be provided if malig is not None
            min_rate: array_like, min proportion of each level in all population.
                      Length should be 4, correspondence of level 1 to 4
            weight: None or str, name of sample weight column in dataset
    Returns:
            results: DataFrame, dataset of each threshold combination, sorted by certain metric
    ----------
    '''   
    
    num = len(data0)
    results = []
    if malig_candidates is None:
        malig_candidates = [None]
    for solid_thres,glass_thres, mix_thres, malig in list(product(solid_candidates, glass_candidates, mix_candidates, malig_candidates)):
        data = split_all_data(data0, var, solid_thres, glass_thres, mix_thres, 'Axis_int')
        flag = 1
        for i in range(4):
            n = len(data[data['Axis_int']==i+1])
            if n < num * min_rate[i]:
                flag = 0
                break 
            # if i == 0:
            #     print(n, n/num, solid_thres,glass_thres, mix_thres, malig)
        if flag == 0:
            continue
        result = judge_single_thres_alldens(data, var, target, solid_thres, glass_thres, mix_thres, malig=malig, malig_col=malig_col, split=False, weight=weight)
        if list(result['woe_l4']) != sorted(list(result['woe_l4']), reverse=True):
            continue
        for key in result:
            if isinstance(result[key], np.ndarray):
                result[key] = np.around(result[key], decimals=2)
        result['thres'] = {'Solid':solid_thres, 'pGGN':glass_thres, 'mGGN':mix_thres, 'malig':malig}
        results.append(result)
    return pd.DataFrame(results).sort_values(sort_key, ascending=False)