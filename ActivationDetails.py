import numpy
import LayerActivation

from marshmallow import Schema, fields

class Activationdetails:

    def __init__(self):
        self.modelName = ''
        self.modeFile = ''
        self.recordName = ''
        self.recordLabel = ''
        self.epoch = ''
        self.activationData = []
        return
    
    def addLayerActivation(self, data):
        self.activationData.append(data)