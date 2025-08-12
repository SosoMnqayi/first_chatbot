import tensorflow as tf

class SavedModel(object):
    def __init__(self, model_path):
        self.model = tf.saved_model.load("saved_model.pb")

    def predict(self, inputs):
        return self.model(inputs)