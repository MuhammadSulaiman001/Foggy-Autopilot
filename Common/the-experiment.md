### The experiment (with project structure explanation)
 
1. **What do we have?**
    1. there are 8000 x 3 = 24000 sunny images, in each (left,center,right) images are taken by the car cameras along with steeringWheel value, so we have a csv file of 8000 rows, each row points to 3 images (features) and 1 label (steeringWheel)
    2. autopilot training code taken from [here](https://github.com/akshaybahadur21/Autopilot)
    3. there is a Weather-GAN model out there, you can give it an image and get the same image with fog, in the degree of fog of your choice
        1. Note1: The code for generating foggy images is in Foggy-Cycle-GAN project, not in this project
        2. Note2: the code for generating foggy images is taken from [here](https://github.com/ghaiszaher/Foggy-CycleGAN)
        3. Note3: The owner of this library does not supply the training data (images), [but I asked him to provide the training models to use them directly, and -thankfully- he did](https://github.com/ghaiszaher/Foggy-CycleGAN/issues/4)
        4. Note4: the generation code is in 'Foggy-Cycle-GAN' project -> 'sami_generator.py' file -> the main method.
2. **Training the first autopilot**
    1. we will train an autopilot that based on sunny images only
    2. the number of images in this training session is 8000 x 3 = 24000 images
    3. the folder of these images is data/IMG_sun_only
    4. the csv of these images is data/driving_log_sun_only.csv, each row in data/driving_log_sun_only.csv contains features (left,center,right) and labels (steeringWheel), 
    5. features will be at models/features_40_sun_only pickle file, and labels will be at models/labels_sun_only pickle file.
    6. the training code is in the main method in ModelTraining.py file.
    7. the final model of this training session is models/MsAutopilot_sun_only.h5
3. We want to train a second autopilot that can recognize the foggy images, we expect the second autopilot to perform better than the first autopilot in a fog-only weather
4. But how to get the foggy images? we will use a pre-existing GAN model to generate these images..
5. **New Data Generation**
    1. we will take these 8000 x 3 sunny images, give them to the Foggy-Cycle-GAN model and get a new 8000 x 3 foggy images, each (left,center,right,steeringWheel) will have a corresponding (left,center,right,steeringWheel) in the foggy data, so the steeringWheel will still have the same value in the foggy data. 
    2. now, we have 8000 x 3 sunny images and a new 8000 x 3 foggy images.
    3. this data is represented as 8000x6 (sunny-left, sunny-center, sunny-right,foggy-left, foggy-center, foggy-right) feature records and 8000x1 (steeringWheel) label records. Remember: the steeringWheel is the same for the sunny and the corresponding foggy triplet.
    4. the generation code is in 'Foggy-Cycle-GAN' project -> 'sami_generator.py' file -> the main method.  
6. **Training data for the second autopilot** 
    1. we will split the (8000x6 features, 8000x1 labels) data in a 80-20 manner, so we will get (6500x6 features, 6500x1 labels).
    2. this (6500x6 features, 6500x1 labels) is the training data for the second autopilot, note that this (6500x6 represents 39000 images) is is larger than the training data of the first autopilot (8000x3 represents 24000 images).
    3. the folder of these (6500x6) images is data/IMG_foggy
    4. the csv of these (6500x6) images is data/driving_log_foggy.csv
    5. each row in data/driving_log_foggy.csv contains 6 features (sunny-left, sunny-center, sunny-right,foggy-left, foggy-center, foggy-right) and 1 label (steeringWheel).
    6. features will be at models/features_40_foggy pickle file, and labels will be at models/labels_foggy pickle file. 
7. **Testing data for both (first and second) autopilots**
    1. the remaining (1500x6 features, 1500x1 labels) represents (1500x3 sunny features, 1500x3 foggy features, 1500x1 labels). 
    2. we will drop the (1500x3 sunny features) and keep the (1500x3 foggy features) with the (1500x1 labels), i.e. we will have (1500x3 foggy features, 1500x1 labels).
    3. **this (1500x3 foggy features, 1500x1 labels) is an unseen fog-only weather images for both first and second autopilots**
    4. the folder of these (1500x3) fog-only images is data/IMG_fog_only
    5. the csv of these (1500x3) images is data/driving_log_fog_only.csv containing 3 features (foggy-left, foggy-center, foggy-right) and 1 label (steeringWheel).
    6. features will be at models/features_40_fog_only pickle file, and labels will be at models/labels_fog_only pickle file.
8. **Training the second autopilot**
    1. we will train the second autopilot on the (6500x6 features, 6500x1 labels) data, i.e. sunny and foggy data
    2. the training code is in ModelTraining.py -> main method
    3. the final model of this training session is models/MsAutopilot_foggy.h5
9. At this point, we have two autopilot models, one is trained on (8000x3 sunny images) and the other is trained on larger data (6500x6 sunny and foggy images), we expect that the second autopilot will perform better than the first one in a fog_only weather conditions (1500x3 unseen foggy images)
10. **Testing both autopilots on the unseen fog-only weather images, (1500x3 foggy features, 1500x1 labels)**
    1. testing code is in ResearchExperiment.py -> main method
    2. We give the same (1500x3 fog_only) images to both autopilots, get the expected 1500x1 steeringWheel values from both, compare them to the correct 1500x1 steeringWheel values and get the evaluation metrics, 
    3. the stats is as follows:
        1. first autopilot (trained on sunny images only): accuracy = 97%
        2. second autopilot (trained on sunny and foggy images): accuracy = 99%
        3. so, the second autopilot performs better than the first one in a fog-only weather conditions!