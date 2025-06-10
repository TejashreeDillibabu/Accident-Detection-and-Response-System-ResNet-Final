from keras.models import model_from_json
import numpy as np

class DetectionModel(object):

    class_nums = ['Major Accident Detected', "Minor Accident Detected", 'No Accident Detected']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_accident(self, img):
        self.preds = self.loaded_model.predict(img)
        return DetectionModel.class_nums[np.argmax(self.preds)], self.preds
