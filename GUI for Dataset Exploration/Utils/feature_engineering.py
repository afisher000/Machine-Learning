import pandas as pd
import numpy as np
import os

class FeatureEngr():
    def __init__(self, main_gui):
        # Attributes
        self.main_gui = main_gui
        self.data_file = 'feature_engineering.csv'
        self.featengr_file = 'feature_engineering_code.txt'
        self.settings_file = 'feature_engineering_settings.csv'
        
    def get_encoding_dict(self, map_str, feature):
        # Create encoding dictionary from map_str
        encode_map = {}
        for entry in map_str.split(','):
            key, value = entry.split('=')
            if key.strip() not in self.data[feature].values:
                raise ValueError(f'Key "{key}" is invalid for given feature')
            encode_map[key.strip()] = float(value.strip())
        return encode_map


    def new_feature(self, transform, feature=None, map_str=None, encodeby=None, code=None, clip_range=None):
        # Create new features for different transforms
        # Logarithm
        if transform=='log1p':
            self.data['log'+feature] = np.log1p(self.data[feature])
            self.settings.loc['log'+feature] = ['log'+feature, False, 'num']
            self.save_featengr_code( 
                feature,
                f'df["log{feature}"]=np.log1p(df["{feature}"])'
            )

        # Binning
        elif transform == 'bin':
            self.data['bin'+feature] = pd.qcut(self.data[feature], 10)
            self.settings.loc['bin'+feature] = ['bin'+feature, False, 'cat']
            self.save_featengr_code( 
                feature,
                f'df["bin{feature}"]=pd.qcut(df["{feature}"], 10)'
            )

        # Dummies
        elif transform == 'dummies':
            # Remove 'Null' dummy
            if self.data[feature].nunique()>11:
                raise ValueError(f'Too many categories for {feature} to create dummies.')
                
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

        # Clipping
        elif transform == 'clip':
            minval, maxval = clip_range
            self.data['clip'+feature] = self.data[feature].clip(lower=minval, upper=maxval)
            feature_type = self.settings.loc[feature, 'feature_type']
            self.settings.loc['clip'+feature] = ['clip'+feature, False, feature_type]

            self.save_featengr_code( 
                feature,
                f'df["clip{feature}"]=df["{feature}"].clip(lower={minval}, upper={maxval})'
            )

        # Drop feature
        elif transform == 'drop':
            self.settings.loc[feature, 'isdropped'] = True

        # Undrop feature
        elif transform == 'undrop':
            self.settings.loc[feature, 'isdropped'] = False

        # Encode
        elif transform == 'encode':
            # Encode by map
            if encodeby=='map':
                try:
                    self.get_encoding_dict(map_str, feature)
                except Exception as e:
                    raise ValueError(str(e))

            # Encode by agg_function
            if encodeby in ['mean', 'median']:
                grouped = self.data.groupby(by=feature)[self.main_gui.target_feature].agg(encodeby)
                encode_map = {}
                for category, median in grouped.items():
                    encode_map[category]=median

            try:
                self.data['enc'+feature] = self.data[feature].map(encode_map).astype('float')
            except Exception as e:
                raise ValueError(str(e))

            self.save_featengr_code( 
                feature, 
                f'df["enc{feature}"] = df["{feature}"].map({encode_map}).astype("float")'
            )
            self.settings.loc['enc'+feature] = ['enc'+feature, False, 'pnum']

        # Code transform
        elif transform == 'code':
            old_features = self.data.columns
            try:
                exec(code)
            except Exception as e:
                raise ValueError(str(e))

            self.save_featengr_code(None, code)

            # Parse new features
            for feature in (set(self.data.columns)-set(old_features)):
                self.settings.loc[feature] = [feature, False, 'none']
                if self.data[feature].dtype == 'object':
                    self.data[feature] = self.data[feature].fillna('Null')
                    self.data[feature] = pd.Categorical( 
                        self.data[feature], np.sort(self.data[feature].unique())
                    )
                    self.settings.loc[feature, 'feature_type'] = 'cat'
                else:
                    if self.data[feature].nunique()<11:
                        self.settings.loc[feature, 'feature_type'] = 'pnum'
                    else:
                        self.settings.loc[feature, 'feature_type'] = 'num'
    
        # Save data and settings
        self.data.to_csv(os.path.join(self.main_gui.directory, self.data_file), index=False)
        self.settings.to_csv(os.path.join(self.main_gui.directory, self.settings_file))
        return

    def save_featengr_code(self, feature, code):
        # Do not save data analysis that uses target_feature
        if feature == self.main_gui.target_feature:
            return
        
        # Write feature engineering code to file
        with open(os.path.join(self.main_gui.directory, self.featengr_file), 'a') as f:
            f.write(code)
            f.write('\n')

    def get_feature_types(self):
        return self.settings.feature_type
    
    def get_discrete_features(self):
        return self.data.columns[self.data.nunique()<0.05*len(self.data)].to_list()
        
    def get_features_by_type(self, feature_types, insert_none=False):
        # Check if list
        if not isinstance(feature_types, list):
            feature_types = [feature_types]

        # Build feature list and sort
        feature_list = []
        for feature_type in feature_types:
            feature_list.extend(self.settings.name[self.settings.feature_type==feature_type].to_list())
        feature_list = sorted(feature_list)

        # Optionally insert 'none'
        if insert_none:
            feature_list.insert(0, 'none')
        return feature_list

    def get_features_by_isdropped(self, isdropped, prefix=False, insert_none=False):
        # Add category prefix
        if prefix:
            full_name = self.settings.feature_type + ' ' + self.settings.name
        else:
            full_name = self.settings.name
        
        # Build feature list and sort
        feature_list = full_name[self.settings.isdropped==isdropped].to_list()
        feature_list = sorted(feature_list)

        # Optionally insert 'none'
        if insert_none:
            feature_list.insert(0, 'none')
        return feature_list

    def create_files(self):
        # Create blank feature_engineering_code 
        with open(os.path.join(self.main_gui.directory, self.featengr_file),'w') as f:
            pass

        # Read train data
        data = pd.read_csv(os.path.join(self.main_gui.directory, 'train.csv'))
        
        # Parse features
        obj_features = data.select_dtypes(['object', 'bool']).columns
        num_features = data.select_dtypes('number').columns
        is_pnum = data[num_features].nunique().lt(11)

        # Convert to object dtypes to Categoricals
        data[obj_features] = data[obj_features].fillna('Null').astype('str')
        for feature in obj_features:
            data[feature] = pd.Categorical(data[feature], np.sort(data[feature].unique()))
        
        # Build featsettings
        settings = pd.DataFrame(index=data.columns)
        settings['name'] = data.columns
        settings['isdropped'] = False
        settings.loc[obj_features, 'feature_type'] = 'cat'
        settings.loc[num_features[is_pnum], 'feature_type'] = 'pnum'
        settings.loc[num_features[~is_pnum], 'feature_type'] = 'num'

        # Save to file
        data.to_csv(os.path.join(self.main_gui.directory, self.data_file), index=False)
        settings.to_csv(os.path.join(self.main_gui.directory, self.settings_file))
        return

    def open_as_categoricals(self, path):
        df = pd.read_csv(path)
        
        # Convert objects to categoricals
        obj_features = df.select_dtypes(['object','bool']).columns
        df[obj_features] = df[obj_features].fillna('Null').astype('str')
        for feature in obj_features:
            df[feature] = pd.Categorical(df[feature], np.sort(df[feature].unique()))
        return df

    def load_files(self):
        # Read or create featengr
        data_path = os.path.join(self.main_gui.directory, self.data_file)
        settings_path = os.path.join(self.main_gui.directory, self.settings_file)

        if not os.path.exists(data_path):
            self.create_files()

        self.data = self.open_as_categoricals(data_path)
        self.settings = pd.read_csv(settings_path, index_col=0)

    def add_residues(self, residues):
        # Add residuals
        self.data['residuals'] = residues
        self.settings.loc['residuals'] = ['residuals', False, 'num']

        # Save to file
        self.data.to_csv(os.path.join(self.main_gui.directory, self.data_file), index=False)
        self.settings.to_csv(os.path.join(self.main_gui.directory, self.settings_file))
        return
