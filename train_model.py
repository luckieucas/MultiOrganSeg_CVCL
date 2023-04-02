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


import argparse
import yaml
import sys
import time
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration_with_multiTask
from nnunet.paths import default_plans_identifier
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.paths import *
from nnunet.training.network_training.nnUNetMultiTrainierV2 import nnUNetMultiTrainerV2
import wandb
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput
def save_config(config):
    if not os.path.exists("config"):
        os.mkdir("config")
    config_file_name = time.strftime("%Y-%m-%d=%H:%M:%S", time.localtime()) + "_{}_train.yaml".format(config['exp'])
    with open(os.path.join("config", config_file_name), "w") as file:
        file.write(yaml.dump(config))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=False, default="train_config.yaml",help='config file path') 
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_file, 'r'))
    save_config(config)
    wandb.init(project="multi_organ_segmentation", name = config['exp'], config=config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
    tasks = config['tasks']
    fold = config['fold']
    do_semi = config['do_semi']
    network = "3d_fullres"
    network_trainer = "nnUNetMultiTrainerV2"
    validation_only = config['validation_only']
    plans_identifier = default_plans_identifier
    find_lr = config['find_lr']

    use_compressed_data = config['use_compressed_data']
    decompress_data = not use_compressed_data

    deterministic = config['deterministic']
    valbest = config['valbest']

    fp32 = config['fp32']
    run_mixed_precision = not fp32

    val_folder = config['val_folder']
    # val_folder = "mk_validation"   #temp_validation
    interp_order = config['interp_order']
    interp_order_z = config['interp_order_z']
    force_separate_z = config['force_separate_z']
    print(config)
    classes_dict = {}
    for i, task in enumerate(tasks):
        if not task.startswith("Task"):
            task_id = int(task)
            task = convert_id_to_task_name(task_id)
        tasks[i] = task

        json_file = join(preprocessing_output_dir,task, "dataset.json")
        classes = []
        with open(json_file) as jsn:
            d = json.load(jsn)
            tags = d['labels']
            for i in tags:
                if not int(i) == 0:#bkg not in tag
                    classes.append(tags[i])
            classes_dict[task] = classes
    if fold == 'all':
        pass
    else:
        fold = int(fold)

    if force_separate_z == "None":
        force_separate_z = None
    elif force_separate_z == "False":
        force_separate_z = False
    elif force_separate_z == "True":
        force_separate_z = True
    else:
        raise ValueError(
            "force_separate_z must be None, True or False. Given: %s" % force_separate_z)

    plans_file, output_folder_names, dataset_directorys, batch_dice, stage, \
        trainer_class = get_default_configuration_with_multiTask(
            network, tasks, network_trainer, plans_identifier, exp=config['exp'])
    if trainer_class is None:
        raise RuntimeError(
            "Could not find trainer class in nnunet.training.network_training")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), \
            "If running 3d_cascade_fullres then your " \
            "trainer class must be derived from " \
            "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class,
                        nnUNetTrainer), "network_trainer was found but is not derived from nnUNetMultiTrainer"

    trainer = trainer_class(plans_file, fold,tasks,tags=classes_dict, output_folder_dict=output_folder_names, dataset_directory_dict=dataset_directorys,
                            batch_dice=batch_dice, stage=0, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision, do_semi=do_semi,wandb=wandb, config=config)
                    

    trainer.initialize(not validation_only)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if config['continue_training']:
                print("continue training")
                trainer.load_latest_checkpoint()
            trainer.run_training() #training
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_latest_checkpoint(train=False)

        trainer.network.eval()

        # predict validation
        for task in tasks:
            print(f"test task: {task}")
            trainer.validate_specific_data(task,save_softmax=config['npz'], validation_folder_name=val_folder, force_separate_z=force_separate_z,overwrite=True,
                            interpolation_order=interp_order, interpolation_order_z=interp_order_z)
    

if __name__ == "__main__":
    main()
