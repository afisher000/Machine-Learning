import pandas as pd
import numpy as np
import os

class Pipe():
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.settings_file = 'pipe_settings.csv'
        self.widget_ptrs = None
        
    def load_files(self):
        # Create file if does not exists
        path = os.path.join(self.file_manager.directory, self.settings_file)
        if os.path.exists(path):
            self.settings = pd.read_csv(path, index_col=0)
            self.settings['impute_by'] = self.settings['impute_by'].astype('str')
        else:
            self.settings = pd.DataFrame(columns=['is_selected','scale_strat','impute_strat', 'impute_by'])
            
            
        
        # Update with new feature engineered features
        features = self.file_manager.featengr.data.select_dtypes('number').columns
        for feature in features:
            if feature not in self.settings.index:
                self.settings.loc[feature] = [False, 'none','none','none']
        self.settings.to_csv(path, index=True)
        
        
        # Create widget pointer dataframe
        self.widget_ptrs = pd.DataFrame(index=features, columns=['checkbox','scaling_menu','imputing_menu','imputeby_menu','fillna_input'])
        return
    
    def update_widgets(self):
        for feature, row in self.widget_ptrs.iterrows():
            is_selected, scale_strat, impute_strat, impute_by = self.settings.loc[feature]
            
            row.checkbox.setChecked(is_selected)
            row.scaling_menu.setCurrentText(scale_strat)
            row.imputing_menu.setCurrentText(impute_strat)
            if impute_strat=='bycategory':
                row.imputeby_menu.setCurrentText(impute_by)
            elif impute_strat=='byvalue':
                row.fillna_input.setText(impute_by)
        return
    
    def get_selected_features(self):
        return self.settings.index[self.settings.is_selected]
    
    def save_settings(self):
        
        for feature, row in self.widget_ptrs.iterrows():
            is_selected = row.checkbox.isChecked()
            scale_strat = row.scaling_menu.currentText()
            impute_strat = row.imputing_menu.currentText()
            if impute_strat=='bycategory':
                impute_by = row.imputeby_menu.currentText()
            elif impute_strat=='byvalue':
                impute_by = row.fillna_input.text()
            else:
                impute_by = 'none'
            self.settings.loc[feature] = [is_selected, scale_strat, impute_strat, impute_by]
        
        path = os.path.join(self.file_manager.directory, self.settings_file)
        self.settings.to_csv(path)

    def get_scaling_dict(self):
        data = {}
        strategies = self.settings[self.settings.is_selected].scale_strat
        for feature, strat in strategies.items():
            if strat=='none':
                continue
            if strat in data:
                data[strat].append(feature)
            else:
                data[strat] = [feature]
        return data  
        
    def get_imputation_dict(self):
        # Build dictionary for imputation implementation
        data = {}
        strategies = self.settings[self.settings.is_selected].impute_strat
        for feature, strat in strategies.items():
            if strat == 'none':
                continue
            
            # Entry in dictinoary depends on strategy
            if strat in ['bycategory', 'byvalue']:
                by_value = self.settings.impute_by[feature]
                entry = (feature, by_value)
            else:
                entry = feature
                
            if strat in data:
                data[strat].append(entry)
            else:
                data[strat] = [entry]
        return data
    
    