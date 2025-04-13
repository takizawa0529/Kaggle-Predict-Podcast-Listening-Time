import os
import pandas as pd

class DataAcquire:
    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data")
        self.files = os.listdir(self.path)

        self.train_files = [file for file in self.files if "train" in file]
        self.test_files = [file for file in self.files if "test" in file]

        self.train_file = os.path.join(self.path, self.train_files[0])
        self.train_extension = os.path.splitext(self.train_file)[1].lower()
        self.test_file = os.path.join(self.path, self.test_files[0])
        self.test_extension = os.path.splitext(self.test_file)[1].lower()


    def judge_extension(self, extension, char):
        return extension == char


    def get_train_data(self):

        if self.judge_extension(self.train_extension, ".tsv"):
            self.df_tmp = pd.read_csv(self.train_file, sep="\t").drop(columns="Unnamed: 0")
        elif self.judge_extension(self.train_extension, ".csv"):
            self.df_tmp = pd.read_csv(self.train_file)  
        return self.df_tmp


    def get_test_data(self):

        if self.judge_extension(self.test_extension, ".tsv"):
            self.df_tmp = pd.read_csv(self.test_file, sep="\t").drop(columns="Unnamed: 0")
        elif self.judge_extension(self.test_extension, ".csv"):
            self.df_tmp = pd.read_csv(self.test_file)
        return self.df_tmp
    

    def get_data(self, switch):
        if switch=="Train":
            df_tmp = self.get_train_data()
        elif switch=="Test":
            df_tmp = self.get_test_data()

        return df_tmp
    

    def space(self, n):
        for _ in range(n):
            print("")


    def fence(self):
        print("="*40)
        self.space(1)


    def fence2(self):
        print("-"*40)


    def get_data_and_columns(self, switch):

        switch_is_not_Train = switch != "Train"
        switch_is_not_Test = switch != "Test"
        if (switch_is_not_Train) and (switch_is_not_Test):
            print("SELECT 'Train' or 'Test' AS switch")
            return 
        else:
            df_tmp = self.get_data(switch)

        print("DATA ACQUIRE:COMPLETE")
        self.fence()

        print("COLUMNS")
        self.fence2()
        print("COLUMN :  D-TYPE :  #NANS")
        for col in df_tmp.columns:
            print(col, ":",  df_tmp[col].dtype, ":", f"{df_tmp[col].isna().sum()} nans")
        self.fence()

        return df_tmp
    

    def get_data_and_description(self, switch):

        switch_is_not_Train = switch != "Train"
        switch_is_not_Test = switch != "Test"
        if switch_is_not_Train and switch_is_not_Test:
            print("SELECT 'Train' or 'Test' AS switch")
            return 
        else:
            df_tmp = self.get_data(switch)

        print("DATA AQCUIRE: COMPLETE")
        self.fence()

        print("DESCRIPTION")
        self.fence2()
        print(df_tmp.describe())
        
        return df_tmp
    
    
    def get_submit_data(self):
        df_tmp = pd.read_csv(os.path.join(self.path, "sample_submission.csv"), header=None)
        return df_tmp
    