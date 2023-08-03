import numpy as np
from sklearn.tree import DecisionTreeClassifier


def sample_weight(level, ca, c0_weight=1):
    if ca == 1:
        return 1
    if ca == 0:
        weight = level * 0.25
        weight = (1.25 - weight)*c0_weight
    return weight
    
def sample_weight_df(df, level_col, patho_col, c0_weight=1):
    """
    Give each sample a weight based on pathological malignancy and expert grading

    Parameters
    ----------
            df: pandas DataFrame, input data
            level col: str, name of pathological malignancy columns in df
            patho_col: str, name of expert grading columns in df

    returns
    ----------
            sw: a list of sample weight for each row in df
    """
    sw = [sample_weight(df[level_col].iloc[i], df[patho_col].iloc[i], c0_weight) for i in range(len(df))]
    df['sample_weight'] = sw
    return df


def divide_nodes(nodes, num_layers, overlap=0, lower_bound=None, upper_bound=None):
    if lower_bound is not None:
        nodes = list(filter(lambda x:x>=lower_bound, nodes))
    if upper_bound is not None:
        nodes = list(filter(lambda x:x<upper_bound, nodes))
    num_each = len(nodes) // num_layers
    nodes = sorted(nodes)
    head = nodes[:num_each+overlap]
    bodies = []
    for i in range(1,num_layers-1):
        body = nodes[(i*num_each-overlap):((i+1)*num_each+overlap)]
        bodies.append(body)
    tail = nodes[-num_each-overlap:]
    return [head] + bodies + [tail]


def thres_from_single_var_decision_tree(df, var, target, max_depth, max_nodes, class_weight, num_layers=3, lower_bound=4, upper_bound=30):
    dct = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_nodes, class_weight=class_weight)
    x = np.array(df[var]).reshape(-1, 1)
    y = df[target]
    sw = df['sample_weight'] if 'sample_weight' in df.columns else None

    dct.fit(X=x, y=y, sample_weight=sw)
    #print(tree.export_text(dct))

    nodes = list(filter(lambda x:x>0, dct.tree_.threshold))
    floors = np.floor(nodes)
    ceils = np.ceil(nodes)

    all_nodes = sorted(list(set(floors)|set(ceils)))
    nodes_layer = divide_nodes(all_nodes, num_layers, 0, lower_bound, upper_bound)
    return dict(zip(range(1, num_layers+1), nodes_layer))


def thres_from_double_vars_decision_tree(df, vars, target, max_depth, max_nodes, class_weight, num_layers=3, lower_bound=4, upper_bound=30):
    dct = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_nodes, class_weight=class_weight)
    x = np.array(df[vars])
    y = df[target]
    sw = df['sample_weight'] if 'sample_weight' in df.columns else None

    dct.fit(X=x, y=y, sample_weight=sw)
    #print(tree.export_text(dct))

    layers = {}
    for i, name in enumerate(['axis', 'solid']):
        nodes = dct.tree_.threshold[dct.tree_.feature==i]
        floors = np.floor(nodes)
        ceils = np.ceil(nodes)

        all_nodes = sorted(list(set(floors)|set(ceils)))
        nodes_layer = divide_nodes(all_nodes, num_layers, 0, lower_bound, upper_bound)
        layers[name] = dict(zip(range(1, num_layers+1), nodes_layer))
    return layers






    



