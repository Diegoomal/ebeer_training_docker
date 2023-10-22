import os
import json
import random
import numpy as np
import tensorflow as tf


def read_data_files(root, folder, sub_folder, file_name):
    with open(os.path.join(root, folder, sub_folder, file_name)) as f:
        files_names = [line.replace('\n', '') for line in f.readlines()]
    return files_names


def read_files_names(root, folder):
    return os.listdir(os.path.join(root, folder))


def read_image(root, folder, file_name) -> None:
    return tf.keras.utils.load_img(
        os.path.join(root, folder, file_name),
        grayscale=False,
        color_mode='rgb',
        target_size=None,
        interpolation='nearest'
    )


def reset_seeds():
    random.seed(123)
    np.random.seed(123)
    tf.random.set_seed(1234)


def get_all_files_in_path_filtered_by_extesion(
        path_dataset: str = "./src/ebeer_dataset/",
        filter_file_ext: list[str] = [".jpg", ".png", "json"]):

    path_files_filtered = [
        os.path.join(path_current_dir, file)
        for path_current_dir, _, files in os.walk(path_dataset)
        for file in files
        if any(
            file.endswith(file_extension)
            for file_extension in filter_file_ext
        )
    ]

    return path_files_filtered


def get_object_from_metadata():
    json_obj = None
    with open('general_metadata.json', 'r') as file_reader:
        json_obj = json.load(file_reader)
    return json_obj