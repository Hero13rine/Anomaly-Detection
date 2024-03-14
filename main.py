import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import sys



##################################
# Choose your model here         #
# You can also use               #
# python main.py <model>         #
# python main.py <algo> <model>  #
##################################
algo = "AircraftClassification"
model = "attention"
#model = "M"

# algo = "FloodingSolver"
# model = "CNN"

# algo = "ReplaySolver"
# model = "HASH"
###################################


if (len(sys.argv) == 2):
    model = sys.argv[1]
elif (len(sys.argv) == 3):
    algo = sys.argv[1]
    model = sys.argv[2]



if (algo == "AircraftClassification"):
    if model == "CNN1":
        import G_Main.AircraftClassification.exp_CNN1 as CNN1
        CNN1.__main__()

    if model == "CNN2":
        import G_Main.AircraftClassification.exp_CNN2 as CNN2
        CNN2.__main__()

    if model == "MSCNN":
        import G_Main.AircraftClassification.exp_MSCNN as MSCNN
        MSCNN.__main__()

    elif model == "LSTM":
        import G_Main.AircraftClassification.exp_LSTM as LSTM
        LSTM.__main__()

    elif model == "Transformer":
        import G_Main.AircraftClassification.exp_Transformer as Transformer
        Transformer.__main__()

    elif model == "Reservoir":
        import G_Main.AircraftClassification.exp_Reservoir as Reservoir
        Reservoir.__main__()
    elif model == 'Unet':
        import  G_Main.AircraftClassification.exp_Unet as Unet
        Unet.__main__()
    elif model == 'attention':
        import G_Main.AircraftClassification.exp_attention as attention
        attention.__main__()


if (algo == "FloodingSolver"):
    if (model == "CNN"):
        import G_Main.FloodingSolver.exp_CNN as CNN
        CNN.__main__()

if (algo == "ReplaySolver"):
    if (model == "HASH"):
        import G_Main.ReplaySolver.exp_HASH as HASH
        HASH.__main__()

