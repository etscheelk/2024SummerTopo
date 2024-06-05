# load some packages
from ctypes import sizeof
import ctypes.util
import plotly.graph_objects as go # oat uses plotly not matplotlib, theres a bit of a learning curve here but it should work fine
import pandas as pd
import oatpy as oat
import numpy as np
import pickle
import dill

import sys

import ctypes





# configuration
# DATA_PATH = 'datasets/CCMathTopologyScavengerHunt/'
if __name__ == "__main__":
    
    DATA_PATH = "datasets/CCMathTopologyScavengerHunt/"
    
    
    # Pull the cloud
    FILE = 'points2.csv'

    dta = np.array(pd.read_csv(DATA_PATH + FILE, header=None)) # oat uses np arrays as coordinate inputs

    print(dta)

    # plotly plotting
    trace = go.Scatter(x=dta[:, 0], y=dta[:, 1], mode='markers')
    fig = go.Figure(trace)
    fig.update_layout(
            width=500, 
            height=500,
            margin=dict(l=20, r=20, t=20, b=20)
        )
    # fig.show()
    
    enclosing = oat.dissimilarity.enclosing_from_cloud(dta) # max fintration radius
    dissimilairty_matrix = oat.dissimilarity.matrix_from_cloud( # distance matrix
        cloud=dta,
        dissimilarity_max=enclosing + 1e-10 # i belive any elements past this are removed (returns a sparse matrix)
    )
    
    # add 1e-10 to elimite some numerical error (greg says to do it)
    factored = oat.rust.FactoredBoundaryMatrixVr( # two functions that do this, idk what the other one is
        dissimilarity_matrix=dissimilairty_matrix,
        homology_dimension_max=1
    )

    # solve homology
    homology = factored.homology( # solve homology
        return_cycle_representatives=True, # These need to be true to be able to make a barcode
        return_bounding_chains=True
    )
    
    print(factored)    
    leng = sys.getsizeof(factored)
    print(leng)
    s = str(factored)
    
    loc = s[-15:-1]
    print(loc)
    # print(int(loc, 16))
    locint = int(loc, 16)
    
    
    char = ctypes.c_char(61)
    print(ctypes.addressof(char))
    
    import _ctypes
    import gi
    
    # pygobject = gi._gobject._PyGObject_API.pygobject_new(locint)

    import help2
    
    obj = help2._PyGObject_Functions.PyGObjectCAPI()
    
    obj2 = obj.to_object(locint)
    
    print(obj)
    
    # pp = ctypes.pointer(int())
    
    

