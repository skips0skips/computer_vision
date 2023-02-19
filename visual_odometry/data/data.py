import json
from pandas import json_normalize
from os import getcwd, path
from yaml import SafeLoader, load
import yaml
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize 
json_object = r'Userdetails.json'

  
with open(json_object) as f:
    d = json.load(f)
    nycphil = pd.json_normalize(d['header'])
    print(nycphil.head(3))
    # works_data = json_normalize(data = d['shots'],
    #                         record_path ='works', 
    #                         meta =['id', 'orchestra', 'programID', 'season'])
    # works_data.head(3)
    #nycphil = json_normalize(d['programs'])
    #nycphil.head(3)



















# path_to_yaml = r'visual_odometry\data\trm.169.007.info.yml'
# stream = open(path_to_yaml, "r")
# docs = yaml.load_all(stream, yaml.FullLoader)
# with open(path_to_yaml, 'r') as f:
#     data = yaml.load(f, Loader=yaml.SafeLoader)

# with open('Userdetails.json', 'w') as f:
#     json.dump(data, f, sort_keys=False)

# for doc in docs:
#     sorted_data = yaml.dump(doc,sort_keys=True)
    
#     print(sorted_data)
