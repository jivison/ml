from keras.models       import model_from_json
from keras.callbacks    import ModelCheckpoint

import data.scripts.contextualize as context

import numpy as np
import json as j
import os

NAME = "languageRecog[16, 32, 32, 4]"

json = None

with open(f"checkpoints/{NAME}|structure.json", "r") as jsonfile:
    json = j.load(jsonfile)

model = model_from_json(str(json))

model.load_weights(f"checkpoints/{NAME}|weights.best.hd5")

model.compile(
    loss='categorical_crossentropy',
    optimizer="sgd",
    metrics=['accuracy'])



print(f"Loaded model: '{NAME}'")

while True:
    try:
        os.system("clear")
        print(f"""
    __                                                ____                              _ __  _                ___    ____
   / /   ____ _____  ____ ___  ______ _____ ____     / __ \___  _________  ____ _____  (_) /_(_)___  ____     /   |  /  _/
  / /   / __ `/ __ \/ __ `/ / / / __ `/ __ `/ _ \   / /_/ / _ \/ ___/ __ \/ __ `/ __ \/ / __/ / __ \/ __ \   / /| |  / /  
 / /___/ /_/ / / / / /_/ / /_/ / /_/ / /_/ /  __/  / _, _/  __/ /__/ /_/ / /_/ / / / / / /_/ / /_/ / / / /  / ___ |_/ /   
/_____/\__,_/_/ /_/\__, /\__,_/\__,_/\__, /\___/  /_/ |_|\___/\___/\____/\__, /_/ /_/_/\__/_/\____/_/ /_/  /_/  |_/___/    ({NAME})
                  /____/            /____/                              /____/                                            



""") 

        word = input("Try me!> ")

        while True:
            if context.validate(word)[0]:
                break
            else:
                print(f"Bad char: '{context.validate(word)[1]}'")

            word = input("Please enter a valid word> ")
        

        Xnew = np.array(context.index(word)).T
        Ynew = model.predict_classes(Xnew)

        print(f"The network guesses: {context.contextualize(Ynew)}")

        print("\nPress enter to try again...")

        input()

    except KeyboardInterrupt:
        print("\nQuitting...")
        quit()
