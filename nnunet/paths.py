'''
Descripttion: 
version: 
Author: Luckie
Date: 2021-06-08 16:15:38
LastEditors: Luckie
LastEditTime: 2021-07-15 15:48:36
'''
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# do not modify these unless you know what you are doing
my_output_identifier = "CAC"
default_plans_identifier = "nnUNetPlansv2.1"
default_data_identifier = 'nnUNet'
default_trainer = "nnUNetTrainerV2"
default_cascade_trainer = "nnUNetTrainerV2CascadeFullRes"

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

base = os.environ['nnUNet_raw_data_base'] if "nnUNet_raw_data_base" in os.environ.keys() else None
preprocessing_output_dir = os.environ['nnUNet_preprocessed'] if "nnUNet_preprocessed" in os.environ.keys() else None
network_training_output_dir_base = os.path.join(os.environ['RESULTS_FOLDER']) if "RESULTS_FOLDER" in os.environ.keys() else None
#base = os.environ['nnUNet_raw_data_base_new'] if "nnUNet_raw_data_base_new" in os.environ.keys() else None #raw data path
# preprocessing_output_dir = os.environ['nnUNet_preprocessed_new'] if "nnUNet_preprocessed_new" in os.environ.keys() else None
# network_training_output_dir_base = os.path.join(os.environ['RESULTS_FOLDER_NEW']) if "RESULTS_FOLDER_NEW" in os.environ.keys() else None
image_validation_output_dir = "/data/liupeng/semi-supervised_segmentation/nnUNetFrame/DATASET/experiment/image/"

if base is not None:
    nnUNet_raw_data = join(base, "nnUNet_raw_splitted")
    nnUNet_cropped_data = join(base, "nnUNet_cropped_data")
    maybe_mkdir_p(nnUNet_raw_data)
    maybe_mkdir_p(nnUNet_cropped_data)
else:
    print("nnUNet_raw_data_base is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read nnunet/paths.md for information on how to set this up properly.")
    nnUNet_cropped_data = nnUNet_raw_data = None

if preprocessing_output_dir is not None:
    maybe_mkdir_p(preprocessing_output_dir)
else:
    print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read nnunet/pathy.md for information on how to set this up.")
    preprocessing_output_dir = None

if network_training_output_dir_base is not None:
    network_training_output_dir = join(network_training_output_dir_base, my_output_identifier)
    maybe_mkdir_p(network_training_output_dir)
else:
    print("RESULTS_FOLDER is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read nnunet/paths.md for information on how to set this "
          "up")
    network_training_output_dir = None

if image_validation_output_dir is not None:
    image_validation_output_dir = join(image_validation_output_dir,my_output_identifier)
    maybe_mkdir_p(image_validation_output_dir)
