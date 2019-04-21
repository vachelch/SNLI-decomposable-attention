import pandas as pd 
import numpy as np

def write_csv(htmls, file):
    csv_file = pd.read_csv('template.csv')

    for i in range(20):
        for j in range(300):
            csv_file.iloc[[i], [j+1]] = "news_%06d"%(htmls[i][j]+1)

    csv_file.to_csv(file, index=False)

res = np.load('res.npy')
htmls = np.argsort(-1*res)
write_csv(htmls, 'preds.csv')
