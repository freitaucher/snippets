import pandas as pd
import numpy as  np
from time import time
from joblib import Parallel, delayed


# init dataframes list:
current_st = pd.DataFrame()
ncols,nrows,ntables=2,10,10000
cols = [ str(i) for i in range(ncols)]
x_list=[ pd.DataFrame( np.random.rand(nrows,ncols) ,columns=cols) for i in range(ntables) ]
y_list=[ pd.DataFrame( np.random.rand(nrows,ncols) ,columns=cols) for i in range(ntables) ]


# auxiliary function:
def aux(x,y):
    return pd.merge_asof( x.sort_values('0'), y.sort_values('0'), on='0',  direction='backward' ) 


"""
# ordinary way:
time_s = time()
for i, (x,y) in enumerate(zip(x_list,y_list)):
    tmp_data = aux(x,y)
    current_st = pd.concat([current_st, tmp_data], ignore_index=True)
time_f = time()
"""


# parallelized
time_s = time()
current_list = pd.concat( Parallel(n_jobs=2)(delayed(aux)(x,y) for x,y in zip(x_list,y_list)), ignore_index=True )
time_f = time()

print('time total: %10.2f sec, per iteration: %10.4f sec' % (time_f-time_s, (time_f-time_s)/len(x_list) ))
