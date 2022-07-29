from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, Lambda
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from Autopilot.src import Utils


class ModelTrainer:
    activation_method = 'relu'

    def __init__(self, pickle_features_path, pickle_labels_path, output_model_path):
        self.pickleFeaturesPath = pickle_features_path
        self.pickleLabelsPath = pickle_labels_path
        self.outputModelPath = output_model_path

    def create_keras_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(40, 40, 1)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation(self.activation_method))
        model.add(MaxPooling2D((2, 2), padding='valid'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation(self.activation_method))
        model.add(MaxPooling2D((2, 2), padding='valid'))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation(self.activation_method))
        model.add(MaxPooling2D((2, 2), padding='valid'))

        model.add(Flatten())
        model.add(Dropout(0.5))

        model.add(Dense(128))

        model.add(Dense(64))
        model.add(Dense(1))

        model.compile(optimizer=Adam(lr=0.0001), loss="mse")
        checkpoint1 = ModelCheckpoint(self.outputModelPath, verbose=1, save_best_only=True)
        callbacks_list = [checkpoint1]

        print("Keras model created successfully..")
        return model, callbacks_list

    def train_my_model(self):
        features, labels = Utils.load_from_pickle(self.pickleFeaturesPath, self.pickleLabelsPath)
        train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                            test_size=0.1)
        train_x = train_x.reshape(train_x.shape[0], 40, 40, 1)
        test_x = test_x.reshape(test_x.shape[0], 40, 40, 1)
        model, callbacks_list = self.create_keras_model()
        print("Training Will Start Now..")
        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=5, batch_size=64,
                  callbacks=callbacks_list)
        model.summary()
        model.save(self.outputModelPath)
        print("Model Trained and saved to" + self.outputModelPath + " successfully")


if __name__ == '__main__':
    # 1. sun-only model generation (MsAutopilot_sun_only.h5)
    """
    outputModelPath = '../models/MsAutopilot_sun_only.h5'
    pickleFeaturesPath = "../models/features_40_sun_only"
    pickleLabelsPath = "../models/labels_sun_only"
    trainer = ModelTrainer(pickleFeaturesPath, pickleLabelsPath, outputModelPath)
    trainer.train_my_model()
    """
    # 2. foggy model generation (MsAutopilot_foggy.h5)
    """
    outputModelPath = '../models/MsAutopilot_foggy.h5'
    pickleFeaturesPath = "../models/features_40_foggy"
    pickleLabelsPath = "../models/labels_foggy"
    trainer = ModelTrainer(pickleFeaturesPath, pickleLabelsPath, outputModelPath)
    trainer.train_my_model()
    """

    # 3. Test (generated model name will end with _delete_me postfix)
    outputModelPath = '../models/MsAutopilot_sun_only_delete_me.h5'
    pickleFeaturesPath = "../models/features_40_sun_only"
    pickleLabelsPath = "../models/labels_sun_only"
    trainer = ModelTrainer(pickleFeaturesPath, pickleLabelsPath, outputModelPath)
    trainer.train_my_model()
