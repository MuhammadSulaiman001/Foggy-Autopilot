from sklearn.model_selection import train_test_split

from Autopilot.src import Utils


class Evaluator:

    def __init__(self, test_data_features_pickle, test_data_labels_pickle, model1, model2):
        self.test_data_features_pickle = test_data_features_pickle
        self.test_data_labels_pickle = test_data_labels_pickle
        self.model1 = model1
        self.model2 = model2

    def evaluate_my_model(self):
        features, labels = Utils.load_from_pickle(self.test_data_features_pickle, self.test_data_labels_pickle)
        _, test_x, _, test_y = train_test_split(features, labels, random_state=0,
                                                test_size=0.1)
        test_x = test_x.reshape(test_x.shape[0], 40, 40, 1)
        print("model1 evaluation started..")
        Utils.print_model_summary(pathToFoggyModel, test_x, test_y)
        print("model2 evaluation started..")
        Utils.print_model_summary(pathToSunOnlyModel, test_x, test_y)


if __name__ == '__main__':
    fogOnlyFeaturesPickle = "../models/features_40_fog_only"
    fogOnlyLabelsPickle = "../models/labels_fog_only"
    pathToFoggyModel = '../models/MsAutopilot_foggy.h5'
    pathToSunOnlyModel = '../models/MsAutopilot_sun_only.h5'
    evaluator = Evaluator(fogOnlyFeaturesPickle, fogOnlyLabelsPickle, pathToFoggyModel, pathToSunOnlyModel)
    evaluator.evaluate_my_model()
    """
    Output will be:
    Test loss for model (models/MsAutopilot_foggy.h5) is: 0.009795374237000942
    Test accuracy for model (models/MsAutopilot_foggy.h5) is: 0.9902046257629991
    Test loss for model (models/MsAutopilot_sun_only.h5) is: 0.02327766828238964
    Test accuracy for model (models/MsAutopilot_sun_only.h5) is: 0.9767223317176104
    """
