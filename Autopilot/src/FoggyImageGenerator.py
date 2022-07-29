# import tensorflow_datasets as tfds
# import os
# from IPython.display import clear_output
# import sys
# import tensorflow as tf
# from lib.models import ModelsBuilder
# from lib.plot import plot_clear2fog_intensity
# from matplotlib import pyplot as plt
# from lib.dataset import DatasetInitializer
# from pathlib import Path
# from lib.tools import create_dir
# from lib.train import Trainer
#
#
# def init(weights_path):
#     models_builder = ModelsBuilder()
#     use_transmission_map = False
#     use_gauss_filter = False
#     use_resize_conv = False
#     use_intensity_for_fog_discriminator = False  # @param{type: "boolean"}
#
#     generator_clear2fog = models_builder.build_generator(use_transmission_map=use_transmission_map,
#                                                          use_gauss_filter=use_gauss_filter,
#                                                          use_resize_conv=use_resize_conv)
#     generator_fog2clear = models_builder.build_generator(use_transmission_map=False)
#
#     discriminator_fog = models_builder.build_discriminator(use_intensity=use_intensity_for_fog_discriminator)
#     discriminator_clear = models_builder.build_discriminator(use_intensity=False)
#
#     trainer = Trainer(generator_clear2fog, generator_fog2clear,
#                       discriminator_fog, discriminator_clear)
#     trainer.configure_checkpoint(weights_path, load_optimizers=False)
#     return generator_clear2fog
#
#
# def generate(generator_clear2fog, inputFilePath, fogFactor, outputFolderPath):
#     imageName = Path(inputFilePath).stem
#     create_dir(outputFolderPath)
#
#     image_clear = tf.io.decode_png(tf.io.read_file(inputFilePath), channels=3)
#     datasetInit = DatasetInitializer(256, 256)
#     image_clear, _ = datasetInit.preprocess_image_test(image_clear, 0)
#     fig = plot_clear2fog_intensity(generator_clear2fog, image_clear, fogFactor)
#     fig.savefig(os.path.join(outputFolderPath, imageName + "_foggy.jpg"),
#                 bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#
#
# if __name__ == '__main__':
#     # 1. init
#     weights_path = "weights"  # folder containing .h5 files (ex generator_clear2fog.h5)
#     generator_clear2fog = init(weights_path)
#
#     # 2. execute
#     outputFolderPath = 'tests/output'
#     inputFolderPath = 'tests/input/'
#     inputFilePath = 'tests/input/sample_clear1.png'  # this will be re-assigned in the for-loop below.
#     fogFactor = 0.2
#     # generate(generator_clear2fog, inputFilePath, fogFactor, outputFolderPath)
#
#     for file in os.listdir(inputFolderPath):
#         inputFilePath = os.path.join(inputFolderPath, file)
#         generate(generator_clear2fog, inputFilePath, fogFactor, outputFolderPath)
#
#     print("Done")
