import numpy as np
import math
import pandas as pd
import os
from PIL import Image
from _Utils import Color  
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})





__icao_db__ = None
def getLabel(CTX, icao, callsign):
    """
    Give the label of an aircraft based on his icao imatriculation
    """
    global __icao_db__
    if __icao_db__ is None:
        labels_file = os.path.join("./A_Dataset/AircraftClassification/labels.csv")
        __icao_db__ = pd.read_csv(labels_file, sep=",", header=None, dtype={"icao24":str})
        __icao_db__.columns = ["icao24", "label"]
        __icao_db__ = __icao_db__.fillna("NULL")


        # merge labels asked as similar
        for label in CTX["MERGE_LABELS"]:
            __icao_db__["label"] = __icao_db__["label"].replace(CTX["MERGE_LABELS"][label], label)

        # to dict
        __icao_db__ = __icao_db__.set_index("icao24").to_dict()["label"]

    if (12 in CTX["MERGE_LABELS"] and callsign.startswith("SAMU")):
        return 12

    if (icao in __icao_db__):
        return __icao_db__[icao]
    
    print("[Warning]", icao, "not found in labels.csv\n\n")
    
    return 0


__airport_db__ = None
__airport_id__ = None
def getAirport(lat, lon):
    """
    give the nearest airport from a lat lon
    """
    global __airport_db__, __airport_id__
    if __airport_db__ is None:
        __airport_db__ = pd.read_csv("./A_Dataset/AircraftClassification/airports.csv", sep=",")
        __airport_id__ = {}

    

    

def resetICAOdb():
    global __icao_db__
    __icao_db__ = None


def getAircraftPosition(CTX, flight):
    # get the aircraft last non zero latitudes and longitudes
    lat = flight[:, CTX["FEATURE_MAP"]["latitude"]]
    lon = flight[:, CTX["FEATURE_MAP"]["longitude"]]
    i = len(lat)-1
    while (i >= 0 and (lat[i] == 0 and lon[i] == 0)):
        i -= 1
    if (i == -1):
        return 0, 0
    return lat[i], lon[i]



def batchPreProcess(CTX, flight, relative_position=False, relative_track=False, random_track=False):
    """
    Additional method of preprocessing after
    batch generation.

    Normalize lat, lon of a flight fragment to 0, 0.
    Rotate the fragment with a random angle to avoid
    biais of the model.
 
    Parameters:
    -----------

    flight: np.array
        A batch of flight fragment

    Returns:
    --------

    flight: np.array
        Same batch but preprocessed


    """
    # Get the index of each feature by name for readability
    FEATURE_MAP = CTX["FEATURE_MAP"]
    lat = flight[:, FEATURE_MAP["latitude"]]
    lon = flight[:, FEATURE_MAP["longitude"]]
    track = flight[:, FEATURE_MAP["track"]]
    groundspeed = flight[:, FEATURE_MAP["groundspeed"]]

    i = 0
    while i < len(lat) and (round(lat[i],  7) == 0 or round(lon[i], 7) == 0):
        i += 1
    if (i == len(lat)):
        return flight
    i -= 1
    while (i >= 0):
        lat[i] = lat[i+1]
        lon[i] = lon[i+1]
        i -= 1

    flight[:, FEATURE_MAP["latitude"]] = lat[:]
    flight[:, FEATURE_MAP["longitude"]] = lon[:]

    last_lat, last_lon = getAircraftPosition(CTX, flight)
    non_zeros_lat_lon = np.logical_or(lat != 0, lon != 0)






    # do not change angle, and rotate the whole bounding box to 0, 0 (not relative just normalizing)
    R = 0
    Y = CTX["BOX_CENTER"][0]
    Z = -CTX["BOX_CENTER"][1]



    if relative_position:
        # R = track[-1]
        Y = last_lat
        Z = -last_lon

    
    if relative_track:
        R = track[-1]

    if random_track:
        R = np.random.uniform(0, 360)

    # Normalize lat lon to 0, 0
    # Convert lat lon to cartesian coordinates
    x = np.cos(np.radians(lon)) * np.cos(np.radians(lat))
    y = np.sin(np.radians(lon)) * np.cos(np.radians(lat))
    z =                           np.sin(np.radians(lat))

    # Normalize longitude with Z rotation
    r = np.radians(Z)
    xz = x * np.cos(r) - y * np.sin(r)
    yz = x * np.sin(r) + y * np.cos(r)
    zz = z

    # Normalize latitude with Y rotation
    r = np.radians(Y)
    xy = xz * np.cos(r) + zz * np.sin(r)
    yy = yz
    zy = -xz * np.sin(r) + zz * np.cos(r)

    # Rotate the fragment with the random angle along X axis
    r = np.radians(R)
    xx = xy
    yx = yy * np.cos(r) - zy * np.sin(r)
    zx = yy * np.sin(r) + zy * np.cos(r)

    # convert back cartesian to lat lon
    lat = np.degrees(np.arcsin(zx))
    lon = np.degrees(np.arctan2(yx, xx))

    # rotate track as well
    track = track - R
    track = np.remainder(track, 360)



    flight[non_zeros_lat_lon, FEATURE_MAP["latitude"]] = lat[non_zeros_lat_lon]
    flight[non_zeros_lat_lon, FEATURE_MAP["longitude"]] = lon[non_zeros_lat_lon]

    # fill empty lat lon with the first non zero lat lon
    first_non_zero_ts = np.argmax(non_zeros_lat_lon)     # (argmax return the first max value)
    first_lat = lat[first_non_zero_ts]
    first_lon = lon[first_non_zero_ts]
    flight[~non_zeros_lat_lon, FEATURE_MAP["latitude"]] = first_lat
    flight[~non_zeros_lat_lon, FEATURE_MAP["longitude"]] = first_lon

    flight[:, FEATURE_MAP["track"]] = track


    if ("timestamp" in FEATURE_MAP):
        timestamp = flight[:, FEATURE_MAP["timestamp"]]
        timestamp = timestamp - timestamp[-1]
        flight[:, FEATURE_MAP["timestamp"]] = timestamp



    return flight
    



def add_noise(flight, label, noise, noised_label_min=0.5):
    """Add same noise to the x and y to reduce bias but 
    mostly to have a more linear output probabilites meaning :
    - avoid to have 1 or 0 prediction but more something
    like 0.3 or 0.7.
    """

    if (noise > 1.0):
        print("ERROR: noise must be between 0 and 1")
        # throw error
        raise ValueError
    if (noise <= 0.0):
        return flight, label

    noise_strength = np.random.normal(0, noise)
    # noise_strength = np.random.uniform(0, noise)

    flight_noise = np.random.uniform(-noise_strength, noise_strength, size=flight.shape)
    flight = flight + flight_noise


    # effective_strength = noise_strength / noise
    # label = label * (1 - effective_strength * (1-noised_label_min))

    return flight, label



"""
UTILITARY FUNCTION FOR MAP PROJECTION
Compute the pixel coordinates of a given lat lon into map.png
"""
def deg2num_int(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = ((lon_deg + 180.0) / 360.0 * n)
    ytile = ((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)




# load image as numpy array
path = "A_Dataset/AircraftClassification/map.png"
img = Image.open(path)
MAP =  np.array(img, dtype=np.float32) / 255.0
def genMap(lat, lon, size):
    """Generate an image of the map with the flight at the center"""

    if (lat == 0 and lon == 0):
        return np.zeros((size, size, 3), dtype=np.float32)


    #######################################################
    # Convert lat, lon to px
    # thoses param are constants used to generate the map 
    zoom = 13
    min_lat, min_lon, max_lat, max_lon = 43.01581, 0.62561,  44.17449, 2.26344
    # conversion
    xmin, ymax = deg2num_int(min_lat, min_lon, zoom)
    xmax, ymin = deg2num_int(max_lat, max_lon, zoom)
    #######################################################

    x_center, y_center = deg2num(lat, lon, zoom)

    x_center = (x_center-xmin)*255
    y_center = (y_center-ymin)*255


    x_min = int(x_center - (size / 2.0))
    x_max = int(x_center + (size / 2.0))
    y_min = int(y_center - (size / 2.0))
    y_max = int(y_center + (size / 2.0))

    d_x_min = x_min
    d_x_max = x_max
    d_y_min = y_min
    d_y_max = y_max

    if (x_min <= 0):
        x_max = size
        x_min = 0

    elif (x_max >= MAP.shape[1]):
        x_max = MAP.shape[1] -1
        x_min = MAP.shape[1] - size -1

    if (y_min <= 0):
        y_max = size
        y_min = 0

    elif (y_max >= MAP.shape[0]):
        y_max = MAP.shape[0] -1
        y_min = MAP.shape[0] - size -1
    
    
    img = MAP[
        y_min:y_max,
        x_min:x_max, :]
    
    if (img.shape[0] != size or img.shape[1] != size):
        print("ERROR: map size is not correct")
        print(MAP.shape)
        print(size)
        print(d_x_min, d_x_max, d_y_min, d_y_max)
        print(x_min, x_max, y_min, y_max)
        print(img.shape)
    
    return img




def inBB(lat, lon, CTX):
    return  lat >= CTX["BOUNDING_BOX"][0][0] \
        and lat <= CTX["BOUNDING_BOX"][1][0] \
        and lon >= CTX["BOUNDING_BOX"][0][1] \
        and lon <= CTX["BOUNDING_BOX"][1][1] \
        

def check_batch(CTX, x, i, t):
    lats = x[i][:, CTX["FEATURE_MAP"]["latitude"]]
    lons = x[i][:, CTX["FEATURE_MAP"]["longitude"]]

    lat = lats[t]
    lon = lons[t]

    if (lat == 0 and lon == 0):
        return False
    


    if (t>0 and lats[t-1] == lats[t] and lons[t-1] == lons[t]):
        return False
    
    if (not inBB(lat, lon, CTX)):
        return False
    
    return True


def pick_an_interesting_aircraft(CTX, x, y, label, n=1, filenames=[]):


    i = -1
    while i == -1 or y[i, label] != 1:
        i = np.random.randint(0, len(x))

    t = None
    tries = 0
    while t == None or not(check_batch(CTX, x, i, t)):

        negative = np.random.randint(0, 100) <= 5
        if (negative):
            t = np.random.randint(CTX["HISTORY"]//2, CTX["HISTORY"]-1)
        else:
            t = np.random.randint(CTX["HISTORY"]-1, len(x[i])-(n-1))
        
        tries += 1
        if (tries > 1000):
            print("Warning: pick_an_interesting_aircraft() failed to find an interesting aircraft in ", filenames[i])
            return pick_an_interesting_aircraft(CTX, x, y, label, n, filenames)



    return i, np.arange(t, t+n)

def to_scientific_notation(number):
    """"
    compute n the value and e the exponent
    """
    if (np.isnan(number)):
        return np.nan, 0
    e = 0
    if (number != 0):
        e = math.floor(math.log10(abs(number)))
        n = number / (10**e)
    else:
        n = 0
    return n, e

def round_to_first_non_zero(number):
    if (number == 0):
        return 0
    n, e = to_scientific_notation(number)
    if (e >= 0):
        return round(number, 1)
    if (e <= -10):
        return round(number, 0)
    
    return round(number, -e)

def round_to_first_non_zero_array(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            array[i][j] = round_to_first_non_zero(array[i][j])
    return array

def print_2D_timeserie(CTX, array, max_len=10):
    """
    shape : [time_step, feature]
    """
    # round numbers to the first non zero decimal
    array = round_to_first_non_zero_array(array)
    array = array.astype(str)
    # remove ending .0
    for i in range(len(array)):
        for j in range(len(array[i])):
            while ("." in array[i][j] and array[i][j].endswith("0")):
                array[i][j] = array[i][j][:-1]
            if (array[i][j].endswith(".")):
                array[i][j] = array[i][j][:-1]
            if (array[i][j].startswith("-0.") or array[i][j] == "-0"):
                array[i][j] = array[i][j][1:]

    max_length = []

    # header
    for i in range(CTX["FEATURES_IN"]):
        # length
        max_feature_length = max([len(array[t][i]) for t in range(len(array))])
        max_length.append(max(max_feature_length, len(CTX["USED_FEATURES"][i])))

    # print header
    for i in range(CTX["FEATURES_IN"]):
        print(Color.GREEN + CTX["USED_FEATURES"][i].ljust(max_length[i]) + Color.RESET, end="|" if i < CTX["FEATURES_IN"]-1 else "\n")


    indexs = np.arange(0, len(array))
    if (len(array) > max_len):
        half = (max_len + 1) // 2
        indexs = np.concatenate([indexs[:half], [-1], indexs[-half:]])

    for t in indexs:
        if (t == -1):
            print("...".rjust(max_length[0]), end="|" if i < CTX["FEATURES_IN"]-1 else "\n")
            continue
        for i in range(CTX["FEATURES_IN"]):
            print(array[t][i].rjust(max_length[i]), end="|" if i < CTX["FEATURES_IN"]-1 else "\n")
    print()



def compute_confidence(y_:np.ndarray):
    return np.max(y_, axis=1) - (np.sum(y_, axis=1) - np.max(y_, axis=1)) / (y_.shape[1]-1)