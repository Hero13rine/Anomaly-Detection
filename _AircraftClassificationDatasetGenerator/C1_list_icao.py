import pandas as pd
import os

from A_parquet2csv import FOLDER

files = os.listdir('./B_csv/'+FOLDER)
files = [file for file in files if file.endswith('.csv')]

icaos = {} 

for file in files:
    try : 
        df = pd.read_csv('./B_csv/'+FOLDER+"/"+ file, dtype={'icao24': str})

        if (df["icao24"][0] in icaos):
            icaos[df["icao24"][0]] += len(df)
        else:
            icaos[df["icao24"][0]] = len(df)

    except Exception as e:
        print(e)
        print("ERROR: " + file + " is corrupted")
        exit(1)
            

# reverse
icaos = {k: v for k, v in sorted(icaos.items(), key=lambda item: item[1], reverse=True)}

# save icao sorted
with open('./labels/A_new_icao.csv', 'w') as f:
    for key in icaos.keys():
        f.write("%s\n" % key)
