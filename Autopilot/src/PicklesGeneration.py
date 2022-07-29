import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
from Autopilot.src import Utils


class PicklesGenerator:
    camerasDelta = 0.2

    def __init__(self, images_folder_name, data_csv_file, no_columns, out_features_path, out_labels_path):
        self.imagesFolderName = images_folder_name
        self.data_csv_file = data_csv_file
        self.NoColumns = no_columns
        self.outFeaturesPath = out_features_path
        self.outLabelsPath = out_labels_path

    def generate_data_pickles(self):
        print("Input Data will be loaded to pickles files..")
        features, labels = self.data_loading()
        features = np.array(features).astype('float32')
        labels = np.array(labels).astype('float32')
        with open(self.outFeaturesPath, "wb") as f:
            pickle.dump(features, f, protocol=4)
        with open(self.outLabelsPath, "wb") as f:
            pickle.dump(labels, f, protocol=4)
        print("Pickles Generated successfully.")

    def data_loading(self):
        logs = []
        features = []
        labels = []
        with open(self.data_csv_file, 'rt') as f:
            reader = csv.reader(f)
            for line in reader:
                logs.append(line)
            _ = logs.pop(0)  # remove headers of csv file..

        for i in range(len(logs)):
            for j in range(self.NoColumns):
                img_path = logs[i][j]
                img_path = features_directory + self.imagesFolderName + (
                    img_path.split(self.imagesFolderName)[1]).strip()
                img = plt.imread(img_path)
                features.append(Utils.preprocess(img))
                if j == 0 or j == 3:
                    labels.append(float(logs[i][self.NoColumns]))
                elif j == 1 or j == 4:
                    labels.append(float(logs[i][self.NoColumns]) + self.camerasDelta)
                elif j == 2 or j == 5:
                    labels.append(float(logs[i][self.NoColumns]) - self.camerasDelta)
        return features, labels


if __name__ == '__main__':
    features_directory = '../../data/'

    # 1. Generate Pickles for sun-only data
    """
    imagesFolderName = 'IMG_sun_only'
    dataCsvFile = '../../data/driving_log_sun_only.csv'
    NoColumns = 3
    outFeaturesPath = "../models/features_40_sun_only"
    outLabelsPath = "../models/labels_sun_only"
    generator = PicklesGenerator(imagesFolderName, dataCsvFile, NoColumns, outFeaturesPath, outLabelsPath)
    generator.generate_data_pickles()
    """

    # 2. Generate Pickles for foggy-only data
    """
    imagesFolderName = 'IMG_foggy'
    dataCsvFile = '../../data/driving_log_foggy.csv'
    NoColumns = 6 # very important, not 3
    outFeaturesPath = "../models/features_40_foggy"
    outLabelsPath = "../models/labels_foggy"
    generator = PicklesGenerator(imagesFolderName, dataCsvFile, NoColumns, outFeaturesPath, outLabelsPath)
    generator.generate_data_pickles()
    """

    # 3. Generate Pickles for fog-only data
    """
    imagesFolderName = 'IMG_fog_only'
    dataCsvFile = '../../data/driving_log_fog_only.csv'
    NoColumns = 3
    outFeaturesPath = "../models/features_40_fog_only"
    outLabelsPath = "../models/labels_fog_only"
    generator = PicklesGenerator(imagesFolderName, dataCsvFile, NoColumns, outFeaturesPath, outLabelsPath)
    generator.generate_data_pickles()
    """

    # 4. Test (generated pickles names will end with _delete_me postfix)
    imagesFolderName = 'IMG_fog_only'
    dataCsvFile = '../../data/driving_log_fog_only.csv'
    NoColumns = 3
    outFeaturesPath = "../models/features_40_fog_only_delete_me"
    outLabelsPath = "../models/labels_fog_only_delete_me"
    generator = PicklesGenerator(imagesFolderName, dataCsvFile, NoColumns, outFeaturesPath, outLabelsPath)
    generator.generate_data_pickles()
