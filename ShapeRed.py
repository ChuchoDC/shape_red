import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

def cargar_datos(csv_file, images_folder, img_size):
    datos = pd.read_csv(csv_file)
    imagenes = []
    landmarks = []

    for _, row in datos.iterrows():
        img_path = os.path.join(images_folder, row['id'])
        img = cv2.imread(img_path)
        if img is None:
            print(f'Advertencia: No se pudo cargar la imagen {row["id"]}.')
            continue

        original_size = (img.shape[1], img.shape[0])
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        img_normalized = img_resized / 255.0
        imagenes.append(img_normalized)

        coords = row[1:].values.astype(np.float32)
        coords[::2] /= original_size[0]
        coords[1::2] /= original_size[1]
        landmarks.append(coords)

    return np.array(imagenes), np.array(landmarks)

imagenes, landmarks = cargar_datos(
    csv_file = 'Archivo_csv',
    images_folder = 'DirectorioDeImagenes',
    img_size = (224, 224)
)
if imagenes.shape[0] != landmarks.shape[0]:
  print("Advertencia, no hay el mismo número de Landmarks e imágenes")

print(f"Imágenes cargadas: {imagenes.shape}")
print(f"Landmarks cargados: {landmarks.shape}")

def modelo(input_shape, num_landmarks):
    model = Sequential([
        Conv2D(32, (7, 7), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(64, (5, 5), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        Conv2D(256, (2, 2), activation='relu', kernel_regularizer=l2(0.01)),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_landmarks * 2, activation='sigmoid')
    ])
    return model

img_size = (224, 224)
input_shape = (img_size[0], img_size[1], 3)
num_landmarks =           # Colocar el número de landmarks con los que se cuenta

modelo = modeloPropuesto00(input_shape, num_landmarks)
modelo.compile(optimizer = Nadam(learning_rate=0.0005), loss="mae")

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)
historial = modelo.fit(imagenes, landmarks, epochs=120, batch_size=4, validation_split=0.25, callbacks=[reduce_lr])

modelo.save('modeloPropuesto.h5')

def predecir_landmarks(modelo, imagenes, output_csv, img_size=(224, 224)):
    predicciones = []

    for img_name in os.listdir(imagenes):
        img_path = os.path.join(imagenes, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f'Advertencia: No se pudo cargar la imagen {img_name}.')
            continue

        original_size = (img.shape[1], img.shape[0])
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = modelo.predict(img, verbose=0)[0]

        pred[::2] *= original_size[0]
        pred[1::2] *= original_size[1]

        predicciones.append([img_name] + pred.tolist())

    columnas = ['id'] + [f'X{i}' for i in range(len(pred) // 2)] + [f'Y{i}' for i in range(len(pred) // 2)]
    resultados = pd.DataFrame(predicciones, columns=columnas)
    resultados.to_csv(output_csv, index=False)
    print(f'Resultados guardados en {output_csv}')


carpeta_imagenes = '/content/drive/MyDrive/NeuralNetworks/Proyecto_Morphometry_NN/PecesSimilares'
output_csv = 'CNN.csv'

modelo = load_model('modeloPropuesto.h5', custom_objects={'mae': MeanAbsoluteError()})
predecir_landmarks(modelo, carpeta_imagenes, output_csv)