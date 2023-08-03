import os
import importlib
import sys
import argparse
from imp import reload
from utils.build_decision_tree import *
from utils.grid_search import *


def load_module_from_disk(pyfile):
    """
    load python module from disk dynamically
    :param pyfile     python file
    :return a loaded network module
    """
    dirname = os.path.dirname(pyfile)
    basename = os.path.basename(pyfile)
    modulename, _ = os.path.splitext(basename)

    need_reload = modulename in sys.modules

    # To avoid duplicate module name with existing modules, add the specified path first.
    os.sys.path.insert(0, dirname)
    lib = importlib.import_module(modulename)
    if need_reload:
        reload(lib)
    os.sys.path.pop(0)

    return lib


class Thresholding():

    def __init__(self, config_file):
        cfg = load_module_from_disk(config_file)
        cfg = cfg.cfg
        self.data = pd.read_csv(cfg.data_list)
        self.save_dir = cfg.save_dir
        self.vars = cfg.axis_colums
        self.target = cfg.pathology_label_column
        self.level = cfg.manual_label_column
        self.mal = cfg.malignancy_prob_column
        self.mal_thres = cfg.malignancy_prob_column
        if self.level is not None and self.target is not None:
            self.data = sample_weight_df(self.data, self.level, self.target)
            self.sw = 'sample_weight'
        else:
            self.sw = None

        self.tree_config = cfg.tree
        self.grid_config = cfg.grading

        self.thres_ranges = None
        self.total_results = None 
    
    def _generate_thres_ranges(self):
        self.solid_data = self.data[self.data.NoduleType==1]
        range_solid = thres_from_single_var_decision_tree(self.solid_data, 
                                                          self.vars[0], self.target, 
                                                          self.tree_config.max_depth, self.tree_config.max_nodes, 
                                                          self.tree_config.class_weight, 
                                                          num_layers=3, 
                                                          lower_bound=self.tree_config.axis_lower_bound, upper_bound=self.tree_config.axis_upper_bound)
        
        self.glass_data = self.data[self.data.NoduleType==2]
        range_glass = thres_from_single_var_decision_tree(self.glass_data, 
                                                          self.vars[0], self.target, 
                                                          self.tree_config.max_depth, self.tree_config.max_nodes, 
                                                          self.tree_config.class_weight, 
                                                          num_layers=2, 
                                                          lower_bound=self.tree_config.axis_lower_bound, upper_bound=self.tree_config.axis_upper_bound)
        
        self.mix_data = self.data[self.data.NoduleType==3]
        ranges_mix = thres_from_double_vars_decision_tree(self.mix_data, 
                                                          self.vars, self.target, 
                                                          self.tree_config.max_depth, self.tree_config.max_nodes, 
                                                          self.tree_config.class_weight, 
                                                          num_layers=2, 
                                                          lower_bound=self.tree_config.axis_lower_bound, upper_bound=self.tree_config.axis_upper_bound)

        self.thres_ranges = {'Solid':range_solid, 'pGGN':range_glass, 'mGGN':ranges_mix}

    
    def _grid_search_combinations(self):
        assert self.thres_ranges is not None

        solid_candids = grid_search_solid(self.solid_data, self.thres_ranges['Solid'], 
                                          self.vars[0], self.target, min_rate=self.grid_config.min_proportion['solid'],
                                          weight=self.sw)
        solid_candids = solid_candids['thres'].iloc[:self.grid_config.num_topK]

        glass_candids = grid_search_glass(self.glass_data, self.thres_ranges['pGGN'], 
                                          self.vars[0], self.target, min_rate=self.grid_config.min_proportion['pGGN'],
                                          weight=self.sw)
        glass_candids = glass_candids['thres'].iloc[:self.grid_config.num_topK]

        mix_candids = grid_search_mix(self.mix_data, self.thres_ranges['mGGN']['axis'], self.thres_ranges['mGGN']['solid'], 
                                      self.vars, self.target, min_rate=self.grid_config.min_proportion['mGGN'])
        mix_candids = mix_candids['thres'].iloc[:self.grid_config.num_topK]

        sort_key = 'iv_sum' if self.mal is None else 'iv_sum_4x'
        total_results = grid_search_all(self.data, self.vars, self.target, solid_candids, glass_candids, mix_candids, 
                                        [self.mal_thres], self.mal, sort_key, self.grid_config.min_proportion['total'],
                                        self.sw)
        self.total_results = total_results.iloc[:self.grid_config.num_topK]

    def run(self):
        self._generate_thres_ranges()
        self._grid_search_combinations()

    def get_result(self):
        assert self.total_results is not None
        top1_thres = self.total_results.thres.iloc[0]
        if self.mal is None:
            top1_thres.pop('malig')
        #print('best threshold: ', top1_thres)
        if self.save_dir is None:
            return top1_thres
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.total_results.to_csv(os.path.join(self.save_dir, 'total_thresholding_results.csv'), index=None)
        return top1_thres


def run_thres_pipeline(config_file):
    th = Thresholding(config_file)
    th.run()
    res = th.get_result()
    print('best threshold: ', res)
    return res


def main():
    long_description = "Gradient Boosting Multidimensional Classification"

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', nargs='?', default='config.py',
                        help='gradient boosting model config file')
    args = parser.parse_args()

    config_file = args.input
    
    run_thres_pipeline(config_file)


if __name__ == '__main__':
    main()

    




        



        
 