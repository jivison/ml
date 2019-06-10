import numpy as np
import json as j
import random

# Which output node corresponds to what language
outputContext = {
    "en" : 0,
    "fr" : 1,
    "de" : 2,
    "la" : 3,
    # "mi" : 4,
    # "es" : 5,
    # "sw" : 6,
    # "et" : 7
}

# Letters
letters = "abcdefghijklmnopqrstuvwxyz"


# Converts letters into numbers
letterDict = {}

# Holds every training example before being turned into...
grandArray = []
    
# This array
outputArray = []

# And this one
inputArray = []

# Turn a string into an array of it's letters' index in letters
def index(word):
    indexarray = [letterDict[letter.lower()] for letter in word]

    # Fill the remaining spots with zeroes (to keep array shape)
    for i in range(0, 16 - len(indexarray)):
        indexarray.append(0)
    return indexarray

def generateOutput(languageCode):
    # There are four output nodes so the array has to be (4, )
    outputarray = [0] * 4

    # Make the right node 1
    outputarray[outputContext[languageCode]] = 1

    return outputarray


def main():

    # Generate the letter conversion dictionary
    for i in range(len(letters)):
        letterDict[letters[i]] = i + 1

    # Load the data
    with open(f"/home/mattecatte/STEM/ml/new/data/scripts/dump/datafile.json", "r") as jsonfile:
        json = j.load(jsonfile)

        # For every word
        for language, words in json.items():
            print(f"Processing language: '{language}'")
            for word in words:
                
                # Appended to an array so that the data can be shuffled (dicts have no order)
                grandArray.append({
                    "output" : generateOutput(language),
                    "input" : index(word),
                    "human symbols" : word,
                    "language" : language
                    })


    # Set the expected output for each input example
    for key in outputContext.keys():
        outputContext[key] = generateOutput(key)

    # Shuffle the data
    random.shuffle(grandArray)

    # Turn the array of key value pairs into two 'linked' arrays
    for i in range(100):
        for element in grandArray:
            outputArray.append(element["output"])
            inputArray.append(element["input"])
        
    print(f"\n~Processed {len(grandArray)} elements~")

    # Return those arrays
    return np.array(inputArray, dtype=float), np.array(outputArray, dtype=float)

