import pandas as pd
import numpy as np
import os

class FeatureEngr():
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.data_file = 'feature_engineering.csv'
        self.featengr_file = 'feature_engineering_code.txt'
        self.settings_file = 'feature_engineering_settings.csv'
        
    def new_feature(self, transform, feature=None, map_str=None, encodeby=None, code=None, clip_range=None):
        if transform=='log1p':
            self.data['log'+feature] = np.log1p(self.data[feature])
            self.settings.loc['log'+feature] = ['log'+feature, False, 'num']
            self.save_featengr_code( 
                feature,
                f'df["log{feature}"]=np.log1p(df["{feature}"])'
            )

        elif transform == 'bin':
            self.data['bin'+feature] = pd.qcut(self.data[feature], 10)
            self.settings.loc['bin'+feature] = ['bin'+feature, False, 'cat']
            self.save_featengr_code( 
                feature,
                f'df["bin{feature}"]=pd.qcut(df["{feature}"], 10)'
            )

        elif transform == 'dummies':
            self.data = pd.concat([ 
                self.data, 
                pd.get_dummies(self.data[feature], prefix=feature)], axis=1
            )
            if 'Null' in self.data[feature].cat.categories:
                self.data = self.data.drop(columns=[feature+'_Null'])
                
            self.save_featengr_code( 
                feature,
                (
                    f'df=pd.concat([df, pd.get_dummies(df["{feature}"], prefix="{feature}")], axis=1)' + '\n'
                    f'if "Null" in df["{feature}"].cat.categories: df=df.drop(columns=["{feature}_Null"])'
                )
            )

            for cat in self.data[feature].unique():
                if cat!='Null':
                    self.settings.loc[f'{feature}_{cat}'] = [f'{feature}_{cat}', False, 'pnum']

        elif transform == 'clip':
            minval, maxval = clip_range
            self.data['clip'+feature] = self.data[feature].clip(lower=minval, upper=maxval)
            feature_type = self.settings.loc[feature, 'feature_type']
            self.settings.loc['clip'+feature] = ['clip'+feature, False, feature_type]

            self.save_featengr_code( 
                feature,
                f'df["clip{feature}"]=df["{feature}"].clip(lower={minval}, upper={maxval})'
            )

        elif transform == 'drop':
            self.settings.loc[feature, 'isdropped'] = True

        elif transform == 'undrop':
            self.settings.loc[feature, 'isdropped'] = False

        elif transform == 'encode':
            # Build encoding map
            if encodeby=='map':
                try:
                    encode_map = {}
                    for entry in map_str.split(','):
                        key, value = entry.split('=')
                        if key.strip() not in self.data[feature].values:
                            raise ValueError(f'Key "{key}" is invalid for given feature')
                        encode_map[key.strip()] = float(value.strip())
                except Exception as e:
                    self.file_manager.data_gui.raise_error('Invalid Map String', f'{e}')
                    return 1
            if encodeby in ['mean', 'median']:
                grouped = self.data.groupby(by=feature)[self.file_manager.target_feature].agg(encodeby)
                encode_map = {}
                for category, median in grouped.items():
                    encode_map[category]=median

                

            try:
                self.data['enc'+feature] = self.data[feature].map(encode_map).astype('float')
            except Exception as e:
                self.file_manager.data_gui.raise_error('Encoding Error', f'{e}')
                return 1

            self.save_featengr_code( 
                feature, 
                f'df["enc{feature}"] = df["{feature}"].map({encode_map}).astype("float")'
            )
            self.settings.loc['enc'+feature] = ['enc'+feature, False, 'pnum']

        elif transform == 'code':
            df = self.data
            old_features = df.columns
            try:
                exec(code)
            except Exception as e:
                self.file_manager.data_gui.raise_error('Code Error', f'{e}')
                return 1
            self.save_featengr_code(None, code)

            # Make parsings feature type a function?
            for feature in (set(df.columns)-set(old_features)):
                self.settings.loc[feature] = [feature, False, 'none']
                if df[feature].dtype == 'object':
                    df[feature] = df[feature].fillna('Null')
                    df[feature] = pd.Categorical(df[feature], np.sort(df[feature].unique()))
                    self.settings.loc[feature, 'feature_type'] = 'cat'
                else:
                    if df[feature].nunique()<11: #TODO make 11 and attribute? Used multiple locations
                        self.settings.loc[feature, 'feature_type'] = 'pnum'
                    else:
                        self.settings.loc[feature, 'feature_type'] = 'num'
    
        self.data.to_csv(os.path.join(self.file_manager.directory, self.data_file), index=False)
        self.settings.to_csv(os.path.join(self.file_manager.directory, self.settings_file))
        return

    def save_featengr_code(self, feature, code):
        # TODO Check if feature is target feature...
        with open(os.path.join(self.file_manager.directory, self.featengr_file), 'a') as f:
            f.write(code)
            f.write('\n')

    def get_feature_types(self):
        return self.settings.feature_type
    
    def get_discrete_features(self):
        return self.data.columns[self.data.nunique()<0.05*len(self.data)].to_list()
        
    def get_features_by_type(self, feature_types):
        # Sort?
        if not isinstance(feature_types, list):
            feature_types = [feature_types]

        feature_list = []
        for feature_type in feature_types:
            feature_list.extend(self.settings.name[self.settings.feature_type==feature_type].to_list())
        return sorted(feature_list)

    def get_features_by_isdropped(self, isdropped, prefix=False):
        # Sort?
        if prefix:
            full_name = self.settings.feature_type + ' ' + self.settings.name
        else:
            full_name = self.settings.name
        feature_list = full_name[self.settings.isdropped==isdropped].to_list()
        return sorted(feature_list)

    def create_files(self):
        # Read and differentiate train data
        data = pd.read_csv(os.path.join(self.file_manager.directory, 'train.csv'))
        obj_features = data.select_dtypes('object').columns
        num_features = data.select_dtypes('number').columns
        is_pnum = data[num_features].nunique().lt(11)

        # Convert to object dtypes to Categoricals
        data[obj_features] = data[obj_features].fillna('Null')
        for feature in obj_features:
            data[feature] = pd.Categorical(data[feature], np.sort(data[feature].unique()))
        
        # Create blank feature_engineering_code 
        with open(os.path.join(self.file_manager.directory, self.featengr_file),'w') as f:
            pass

        # Build featsettings
        settings = pd.DataFrame(index=data.columns)
        settings['name'] = data.columns
        settings['isdropped'] = False
        settings.loc[obj_features, 'feature_type'] = 'cat'
        settings.loc[num_features[is_pnum], 'feature_type'] = 'pnum'
        settings.loc[num_features[~is_pnum], 'feature_type'] = 'num'


        # Save to file
        data.to_csv(os.path.join(self.file_manager.directory, self.data_file), index=False)
        settings.to_csv(os.path.join(self.file_manager.directory, self.settings_file))
        return

    def open_as_categoricals(self, path):
        df = pd.read_csv(path)

        obj_features = df.select_dtypes('object').columns
        df[obj_features] = df[obj_features].fillna('Null')
        for feature in obj_features:
            df[feature] = pd.Categorical(df[feature], np.sort(df[feature].unique()))
        return df

    def load_files(self):
        # Read or create featengr
        data_path = os.path.join(self.file_manager.directory, self.data_file)
        settings_path = os.path.join(self.file_manager.directory, self.settings_file)

        if not os.path.exists(data_path):
            self.create_files()

        self.data = self.open_as_categoricals(data_path)
        self.settings = pd.read_csv(settings_path, index_col=0)

    def add_residues(self, residues):
        self.data['residuals'] = residues
        self.settings.loc['residuals'] = ['residuals', False, 'num']
        return
