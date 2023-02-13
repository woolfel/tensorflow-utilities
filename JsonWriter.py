import ActivationDetails as ad
import json

def writeActivationDetails(result: ad.Activationdetails):
    jsonstring = '{'
    jsonstring += '"modelName":"' + result.modelName + '",'
    jsonstring += '"modelFile":"' + result.modeFile + '",'
    jsonstring += '"recordName":"' + result.recordName + '",'
    jsonstring += '"recordLabel":"' + str(result.recordLabel) + '",'
    jsonstring += '"epoch":"' + result.epoch + '",'
    jsonstring += '"activationData":['
    for i in range(len(result.activationData)):
        layeract = result.activationData[i]
        if i > 0:
            jsonstring += ','
        jsonstring += '{'
        jsonstring += '"layerName":"' + layeract.layerName + '",'
        jsonstring += '"layerIndex":' + str(layeract.layerIndex) + ','
        jsonstring += '"layerType":"' + layeract.layerType + '",'
        jsonstring += '"shape":"' + layeract.shape + '",'
        jsonstring += '"activations":' + json.dumps(layeract.activations.tolist())
        jsonstring += '}'
    jsonstring += ']}'
    return jsonstring