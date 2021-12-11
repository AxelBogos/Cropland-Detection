import pandas as pd 
import numpy as np

def agg_data(df: pd.DataFrame, agg_size: int = 4, train = True, verbose = True,  save = False) -> None:     
    if train: 
        labels = df['LABELS']
        df.drop(columns = 'LABELS', inplace = True)

    assert(df.shape[1] == 216)
    assert(12 % agg_size == 0)

    sensors = np.arange(18)

    data = []
    col_names = []

    for sensor in sensors:
        curr_count = 0
        curr_sum = np.zeros_like(df.iloc[:, 0])
        for i in range(sensor, df.shape[1], 18): 
            curr_sum += df.iloc[:, i]
            curr_count += 1 

            if(curr_count % agg_size == 0): 
                col_name = df.columns[i][:-3] + 'g' + str(curr_count//agg_size)
                col_names.append(col_name)
                data.append(pd.Series(curr_sum / agg_size))
                curr_sum = np.zeros_like(df.iloc[:, 0])

    agg_df = pd.concat(data, axis = 1, keys= col_names)
    
    if train: 
        agg_df['LABELS'] = labels

    if verbose: 
        if train: 
            print(f"Mode : train")
        else: 
            print(f"Mode : test")
        print(f"Original data shape   : {df.shape}")
        print(f"Aggregated data shape : {agg_df.shape}")

    if save: 
        if train: 
            filename = "train_" + str(agg_size) + ".csv"
            agg_df.to_csv("./data/processed/" + filename)
        else: 
            filename = "test_" + str(agg_size) + ".csv"
            agg_df.to_csv("./data/processed" + filename)

train = pd.read_csv("./data/orig/train.csv", index_col= 0)
test = pd.read_csv("./data/orig/test.csv", index_col= 0)

agg_data(train, agg_size = 4, train= True, verbose= True, save = True)
agg_data(test, agg_size = 4, train= False, verbose= True, save = True)