import pickle
#import pandas as pd
import numpy as np
with open("ep_rews.pickle", "rb") as w:
    data = pickle.load(w)
    print(len(data))

pd.DataFrame(np.array(data)).to_csv("rewards_data.csv")

print("hello")