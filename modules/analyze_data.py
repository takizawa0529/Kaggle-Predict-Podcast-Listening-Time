import numpy as np
import pandas as pd
import seaborn as sns
import japanize_matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import k_means


class analyze_data:
    def __init__(self, df):
        self.df = df


    def scatter(self, col1, col2):
        plt.scatter(self.df[col1], self.df[col2])
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(axis="both")
        plt.show()
    
    
    def scatter_hue(self, col1, col2, hue_col):
        
        for element in np.sort(self.df[hue_col].unique()):
            df_tmp = self.df[self.df[hue_col]==element]
            plt.scatter(df_tmp[col1], df_tmp[col2], label=f"{hue_col} = {element}")

        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(axis="both")
        plt.legend()
        plt.show()
