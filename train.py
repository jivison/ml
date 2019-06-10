# Keras imports
from keras.models       import Sequential
from keras.layers       import Dense
from keras.callbacks    import ModelCheckpoint

import numpy as np
import data.scripts.generateMatrix 
from json import dump

NAME = "languageRecog[16, 32, 32, 4]"

x_train, y_train = data.scripts.generateMatrix.main()

model = Sequential()

model.add(Dense(units=16, activation='relu', input_shape=(16, )))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=4, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer="sgd",
    metrics=['accuracy'])

weightsfile_path = f"checkpoints/{NAME}|weights.best.hd5"
structurefile_path = f"checkpoints/{NAME}|structure.json"

with open(structurefile_path, "w") as structurefile:
    dump(model.to_json(), structurefile)

checkpoint = ModelCheckpoint(weightsfile_path, monitor="acc", verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]

model.fit(x_train, y_train, epochs=2000, batch_size=32, callbacks=callback_list)
