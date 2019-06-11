# Keras imports
from keras.models       import model_from_json
from keras.callbacks    import ModelCheckpoint

# Custom import, provides various functions to switch between human and network readable
import data.scripts.contextualize as context

# Numpy for arrays
import numpy as np

# Json for reading the network architecture
import json as j

# To clear the screen
import os

# Name of the AI that will be loaded
NAME = "languageRecog3.2[16, 64, 64, 4]"

print(f"Loading model: '{NAME}'")

# Intialize the json variable
json = None

# Read the architecture jsonfile
with open(f"checkpoints/{NAME}|structure.json", "r") as jsonfile:
    json = j.load(jsonfile)

# Load the json into the model
model = model_from_json(str(json))

# Load the hd5 weights model
model.load_weights(f"checkpoints/{NAME}|weights.best.hd5")

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer="sgd",
    metrics=['accuracy'])


while True:
    try:
        os.system("clear")
  
#   B)
        print(f"""
    __                                                ____                              _ __  _                ___    ____
   / /   ____ _____  ____ ___  ______ _____ ____     / __ \___  _________  ____ _____  (_) /_(_)___  ____     /   |  /  _/
  / /   / __ `/ __ \/ __ `/ / / / __ `/ __ `/ _ \   / /_/ / _ \/ ___/ __ \/ __ `/ __ \/ / __/ / __ \/ __ \   / /| |  / /  
 / /___/ /_/ / / / / /_/ / /_/ / /_/ / /_/ /  __/  / _, _/  __/ /__/ /_/ / /_/ / / / / / /_/ / /_/ / / / /  / ___ |_/ /   
/_____/\__,_/_/ /_/\__, /\__,_/\__,_/\__, /\___/  /_/ |_|\___/\___/\____/\__, /_/ /_/_/\__/_/\____/_/ /_/  /_/  |_/___/    ({NAME})
                  /____/            /____/                              /____/                                            



""") 

        # Get the user's input word
        word = input("Try me!> ")

        while True:

            # Check the validity of the word
            if context.validate(word)[0]:
                break
            else:
                # Let the user know where the problem is 
                print(f"Bad char: '{context.validate(word)[1]}'")

            # Scold the user for being bad
            word = input("Please enter a valid word> ")

        # Judo slice
        word = word[:16]

        # Machine readify the user's input
        Xnew = np.array(context.index(word), dtype=float).T

        # Get the network's guess
        Ynew = model.predict_classes(Xnew)

        print(f"The network guesses: {context.contextualize(Ynew)}")

        print("\nPress enter to try again...")

        input()


    except KeyboardInterrupt:
    # Smoother quit
        print("\nQuitting...")
        quit()
