from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True}) # to automatically fit long axis ticklabels

import numpy as np
import pandas as pd
import seaborn as sns
class GUI_Figure():
    def __init__(self, GUI, layout):
        self.GUI = GUI
        self.canvas = FigureCanvas(Figure(figsize=(3,3)))
        layout.addWidget(NavigationToolbar(self.canvas, self.GUI))
        layout.addWidget(self.canvas)
    
    def reset_figure(self, ncols):
        self.canvas.figure.clear()
        self.ax = self.canvas.figure.subplots(ncols=ncols)
        return
    

    def custom_plot(self, data, plot_type, target_feature, n_features):
        self.reset_figure(ncols=1)
        if plot_type == 'correlation':
            ( 
                data.corrwith(data[target_feature], numeric_only=True)
                .abs().sort_values(ascending=False)[1:n_features+1]
                .plot.bar(xlabel='Feature', ylabel='Correlation', title=f'Target={target_feature}', ax=self.ax)
            )
        elif plot_type == 'null percentage':
            ( 
                (data.isna().sum()+data.apply(lambda x: sum(x=='Null')))
                .sort_values(ascending=False).divide(len(data))[:n_features]
                .plot.bar(xlabel='Feature', ylabel='Null Percentage', ax=self.ax)
            )
        elif plot_type == 'largest category':
            ( 
                data.apply(lambda x: x.value_counts().iloc[0]/len(x))
                .sort_values(ascending=False)[:n_features]
                .plot.bar(xlabel='Feature', ylabel='Largest Cat. Fraction', ax=self.ax)
            )
        elif plot_type == 'number of categories':
            ( 
                data.select_dtypes(exclude='number').nunique()
                .sort_values(ascending=False)[:n_features]
                .plot.bar(xlabel='Feature', ylabel='Number of Categories', ax=self.ax)
            )
        self.draw()
        
    def boxenplot_and_piechart(self, data, x, y, hue, agg_fcn=None, switch_axes=False, median_ordering=False):
        self.reset_figure(ncols=2)
        if switch_axes:
            x, y = y, x
            
        if median_ordering:
            data = data.copy()
            median_ordering = data.groupby(by=x)[y].median().sort_values().index.tolist()
            data[x] = data[x].cat.reorder_categories(median_ordering)
            
        if data[x].nunique()>0.05*len(data):
            return

        if agg_fcn is None:
            sns.boxenplot(data=data, x=x, y=y, hue=hue, ax=self.ax[0])
        else:
            sns.barplot(data=data.dropna(subset=[x,y]), x=x, y=y, hue=hue, estimator=agg_fcn, ax=self.ax[0])

        self.ax[0].tick_params(axis='x', rotation=90)
        data[x].value_counts(dropna=False).sort_index().plot.pie(autopct='%.0f%%', ax=self.ax[1])
        self.draw()
        return 

    
    def histplot_and_piechart(self, data, x, hue, stat=None):
        # Add check for number of unique in x?
        self.reset_figure(ncols=2)

        if data[x].nunique()>0.05*len(data):
            print('save yourself!')
            return
        if hue is not None:
            if data[hue].nunique()>0.05*len(data):
                print('save yourself!')
                return
        sns.histplot(data=data, x=x, hue=hue, stat='count', multiple='stack', kde=False, ax=self.ax[0])
        self.ax[0].tick_params(axis='x', rotation=90)
        data[x].value_counts(dropna=False).sort_index().plot.pie(autopct='%.0f%%', ax=self.ax[1])
        self.draw()
        return
        
    def histplot(self, data, x, hue):
        self.reset_figure(ncols=1)
        sns.histplot(data=data, x=x, hue=hue, multiple='stack', kde=True, ax=self.ax)
        self.draw()
        return
    
    def scatterplot(self, data, x, y, hue, size, style):
        self.reset_figure(ncols=1)
        sns.scatterplot(data=data, x=x, y=y, hue=hue, size=size, style=style, 
                        ax=self.ax)
        self.draw()
        return
        
    def draw(self):
        self.canvas.draw()