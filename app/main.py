import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # necessario para o acesso a GPU
import json
import numpy as np
from PIL import Image
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, MaxPooling2D)

from configs import WIDTH, HEIGHT, CHANNELS, SIZE
from data_augmentation import execute_data_augmentation_operation_on_image_by_path
from util import (read_image, reset_seeds, get_all_files_in_path_filtered_by_extesion, get_object_from_metadata)


if __name__ == "__main__":

  PATH_DATASET = '../ebeer_dataset'

  print('1) Verifica o acesso a placa de video')
  print(tf.config.list_physical_devices('GPU'))



  print('2) Testa se baixou o dataset')
  img = Image.open(PATH_DATASET + '/0/0.jpg')
  print(f'ebeer_dataset/0/0.jpg --> width: {img.size[0]} x height {img.size[1]} (pixels)')



  print('3) Acessa os arquivos de metadados e transforma em um arquivo geral')

  paths_metadata = get_all_files_in_path_filtered_by_extesion(PATH_DATASET, ["json"])

  json_content_arr = []
  for path_metadata in paths_metadata:
    with open(path_metadata, 'r') as file:
      file_content = json.load(file)
      json_content_arr.append(file_content)

  json_content_arr = sorted(list(json_content_arr), key=lambda x: x["path"])

  with open("general_metadata.json", "w") as file_writer:
    json.dump(json_content_arr, file_writer)



  print('4) Inicia o processo de data augmentation')

  paths_dataset_images = get_all_files_in_path_filtered_by_extesion(PATH_DATASET, [".jpg", ".png"])

  with tf.device('/gpu:0'):
    for path_file in paths_dataset_images:
      execute_data_augmentation_operation_on_image_by_path(path_file)

  print("execute_data_augmentation -> len(imgs):", len(paths_dataset_images))



  print('5) Estrutura dataset para treino')

  json_obj = get_object_from_metadata()

  N_NEURONS_OUT = len(json_obj)

  list_train_data_X = []
  list_train_data_y = []

  for i in range(N_NEURONS_OUT):

    paths_dataset_images = get_all_files_in_path_filtered_by_extesion(f"{PATH_DATASET}/{i}", [".jpg", ".png"])

    train_data = [
      np.array(image).astype('float32')/255 for image in
      [
        read_image('', '', image_name).resize(SIZE)
        for image_name in paths_dataset_images
      ]
    ]

    list_train_data_X.append(train_data)
    list_train_data_y.append(np.ones(len(train_data)) * i)



  print('6) cria modelo de rede neural')

  reset_seeds()

  model = Sequential()

  # Extração de caracteristicas
  model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, CHANNELS)))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))

  # Achatamento
  model.add(Flatten())

  # classificadores
  model.add(Dense(128, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(N_NEURONS_OUT, activation='softmax'))

  learning_rate = 1e-4

  model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=[
      'accuracy',
      tf.keras.metrics.Precision(),
      tf.keras.metrics.Recall()
    ]
  )



  print('7) Treinamento do modelo')

  X_train, X_test, y_train, y_test = train_test_split(
    np.concatenate(list_train_data_X),
    to_categorical(np.concatenate(list_train_data_y)),
    test_size=0.3,
    random_state=42
  )
  
  reset_seeds()

  with tf.device('/gpu:0'):
    hist = model.fit(X_train, y_train, epochs=25, validation_split=0.2)

  model.evaluate(X_test, y_test)

  model.save('trained_model.h5')



  print('8) Predição')

  def predicao(filename):
    image_original = tf.keras.utils.load_img(
      PATH_DATASET + f'/tests/{filename}',
      grayscale=False,
      color_mode='rgb',
      target_size=None,
      interpolation='nearest'
    )

    image_prepared = np.expand_dims(image_original.resize(SIZE), axis=0)

    predicted = model.predict(image_prepared)

    n_pos = predicted.argmax(axis=-1)[0]

    json_obj = get_object_from_metadata()

    dict_labels_index = {i: json_obj[i] for i in range(len(json_obj))}

    print("--->:", dict_labels_index[n_pos]['name'])
  
  predicao(filename='0.jpg')
  predicao(filename='1.png')
  predicao(filename='2.jpg')