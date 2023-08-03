import os
import importlib
import sys
import argparse
from imp import reload
from utils.gradient_boosting import *


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


class GBClassification():
    def __init__(self, config_file):
        cfg = load_module_from_disk(config_file)
        cfg = cfg.cfg
        self.data = pd.read_csv(cfg.data_list) if 'csv' in cfg.data_list else pd.read_excel(cfg.data_list)
        self.save_dir = cfg.save_dir

        self.image_col = cfg.image_column
        self.tag_col = cfg.pathology_label_column
        self.sex_col = cfg.sex_column
        self.cli_cols = cfg.clinical_columns
        self.follow_cols = cfg.followup_columns
        self.descrete_cols = cfg.descrete_columns
        self.total_cols = self.cli_cols + [self.sex_col] + self.follow_cols

        self.model = cfg.model

        self.coef = None
        self.intercept = None
        self.feature_names = None


    def _feature_fitting_pipeline(self):
        df_pred, feature_names, coef, intercept = boosting_onestep_coef_bagging(self.data, self.sex_col,
                                                    self.image_col, self.cli_cols, self.tag_col, self.descrete_cols,
                                                    1, self.model.type, 
                                                    self.model.clinic_bagging_num, self.model.clinic_bagging_frac,
                                                    self.model.loss_func, image_func=f, 
                                                    fillna=self.model.fillna, fillval=self.model.fillval, 
                                                    weight_pos=self.model.weight_pos, 
                                                    dummy=self.model.dummy, use_intercept=self.model.use_intercept)
        lasso_f = boosting_onestep_coef(df_pred, 'pred_cli', self.follow_cols, self.tag_col, self.descrete_cols,
                                        self.model.type, self.model.loss_func, f, 
                                        fillna=self.model.fillna, fillval=self.model.fillval, 
                                        weight_pos=self.model.weight_pos_follow, dummy=self.model.dummy,
                                        use_intercept=self.model.use_intercept)
        
        coef0 = lasso_f.coef_.reshape(-1,1)
        feature_names.extend(list(lasso_f.feature_names_in_))
        self.coef = np.concatenate((coef,coef0))
        self.intercept = intercept + lasso_f.intercept_
        self.feature_names = feature_names
    
    def data_inference(self, data=None):
        if data is None:
            data = self.data
        df_pred = boost_lasso_predict(data, self.image_col, self.total_cols, self.descrete_cols,
                                      self.coef, self.intercept, self.feature_names, 
                                      fillna=True, fillval=0, dummy=self.model.dummy,pred_name='pred_multiD')[0]
        return df_pred

    
    def set_linear_model(self, coef, intercept, feature_names=None):
        if self.coef is not None or self.intercept is not None:
            print('Existing linear model will be reset! ')
        else:
            print('Setting linear model. ')
        if len(coef.shape) == 1:
            coef = coef.reshape(-1, 1)
        assert len(coef.shape) == 2
        self.coef = coef
        self.intercept = intercept
        self.feature_names = feature_names

    def run(self):
        self._feature_fitting_pipeline()

    def get_results(self):

        if 'pred_multiD' not in list(self.data.columns):
            df_pred = self._data_inference()
        else:
            df_pred = self.data

        if self.save_dir is None:
            return dict(zip(('feature_name', 'coefficient', 'intercept'), (self.feature_names, self.coef, self.intercept))), df_pred
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        
        df_pred.to_csv(os.path.join(self.save_dir, 'GBClassification_results.csv'), index=None)
        return dict(zip(('feature_name', 'coefficient', 'intercept'), (self.feature_names, self.coef, self.intercept))), df_pred


def run_gb_pipeline(config_file_path, test_data=None):
    gb = GBClassification(config_file_path)
    gb.run()
    model, data_new = gb.get_results()
    if test_data is None:
        return model, data_new
    test_data = pd.read_csv(test_data) if 'csv' in test_data else pd.read_excel(test_data)
    test_result = gb.data_inference(test_data)
    return model, data_new, test_result

def main():
    long_description = "Gradient Boosting Multidimensional Classification"

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', nargs='?', default='config.py',
                        help='gradient boosting model config file')
    parser.add_argument('-t', '--test', default=None, 
                        help='Optional test data file path')
    args = parser.parse_args()

    config_file = args.input
    test_file = args.test
    
    run_gb_pipeline(config_file, test_file)


if __name__ == '__main__':
    main()



       
        
                                                

