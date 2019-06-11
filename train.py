# Keras imports
from keras.models       import Sequential
from keras.layers       import Dense
from keras.callbacks    import ModelCheckpoint

# Numpy for arrays
import numpy as np

# Custom import, generates machine-understandable matrixes from raw data
import data.newScripts.generateMatrix 

# Used for saving the model's architecture
from json import dump

# The AI's name (and structure) for saving
NAME = "languageRecog3.3[16, 64, 64, 4]"

# The training data, where X is input data and y is the expected output
x_train, y_train = data.newScripts.generateMatrix.main()

# Initialize the model as a Sequential one
model = Sequential()

# Add the layers to the network              input shape is 16 because our words have a max length of 16
model.add(Dense(units=16, activation='relu', input_shape=(16, )))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=4, activation='sigmoid'))

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer="sgd",
    metrics=['accuracy'])

# Where the model will be 'saved'
weightsfile_path = f"checkpoints/{NAME}|weights.best.hd5"
structurefile_path = f"checkpoints/{NAME}|structure.json"

# Dump the structure to the approriate file
with open(structurefile_path, "w") as structurefile:
    dump(model.to_json(), structurefile)

# Callback which saves the model
checkpoint = ModelCheckpoint(weightsfile_path, monitor="acc", verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]

# Trains the AI!
model.fit(x_train, y_train, epochs=20000, batch_size=32, callbacks=callback_list)
