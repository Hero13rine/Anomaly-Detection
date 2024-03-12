

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.Reservoir.modules import RC_model
import os
import numpy as np


class Model(AbstactModel):
    """
    Convolutional neural network model for 
    aircraft classification based on 
    recordings of ADS-B data fragment.

    Parameters:
    ------------

    CTX: dict
        The hyperparameters context


    Attributes:
    ------------

    name: str (MENDATORY)
        The name of the model for mlflow logs
    
    Methods:
    ---------

    predict(x): (MENDATORY)
        return the prediction of the model

    compute_loss(x, y): (MENDATORY)
        return the loss and the prediction associated to x, y and y_

    training_step(x, y): (MENDATORY)
        do one training step.
        return the loss and the prediction of the model for this batch

    visualize(save_path):
        Generate a visualization of the model's architecture
    """

    name = "Reservoir"

    def __init__(self, CTX:dict):
        """ 
        Generate model architecture
        Define loss function
        Define optimizer
        """


        # load context
        self.CTX = CTX
        
        self.model =  RC_model(
                        reservoir=None,     
                        n_internal_units=self.CTX['n_internal_units'],
                        spectral_radius=self.CTX['spectral_radius'],
                        leak=self.CTX['leak'],
                        connectivity=self.CTX['connectivity'],
                        input_scaling=self.CTX['input_scaling'],
                        noise_level=self.CTX['noise_level'],
                        circle=self.CTX['circ'],
                        n_drop=self.CTX['n_drop'],
                        bidir=self.CTX['bidir'],
                        dimred_method=self.CTX['dimred_method'], 
                        n_dim=self.CTX['n_dim'],
                        mts_rep=self.CTX['mts_rep'],
                        w_ridge_embedding=self.CTX['w_ridge_embedding'],
                        readout_type=self.CTX['readout_type'],            
                        w_ridge=self.CTX['w_ridge'],              
                        mlp_layout=self.CTX['mlp_layout'],
                        num_epochs=self.CTX['num_epochs'],
                        w_l2=self.CTX['w_l2'],
                        nonlinearity=self.CTX['nonlinearity'], 
                        svm_gamma=self.CTX['svm_gamma'],
                        svm_C=self.CTX['svm_C']
                        )
        
    def predict(self, x):
        """
        Make prediction for x 
        """
        fake_y = np.zeros((x.shape[0], self.CTX["FEATURES_OUT"]))
        
        return self.compute_loss(x, fake_y)[1]

    def compute_loss(self, x, y):
        """
        Make a prediction and compute the loss
        that will be used for training
        """
        acc, f1, out = self.model.test(x, y)
        
        return f1, out

    def training_step(self, x, y):
        """
        Do one forward pass and gradient descent
        for the given batch
        """
        time = self.model.train(x, y)

        return self.compute_loss(x, y)



    def getVariables(self):
        """
        Return the variables of the model
        """
        return None

    def setVariables(self, variables):
        """
        Set the variables of the model
        """
        pass
