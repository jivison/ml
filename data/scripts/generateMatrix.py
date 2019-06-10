import numpy as np
import json as j
import random

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

letters = "abcdefghijklmnopqrstuvwxyz"


# Converts letters into numbers
letterDict = {}


# Holds every training example before being turned into...
grandArray = []
    
# This array
outputArray = []

# And this one
inputArray = []


def index(word):
    indexarray = [letterDict[letter.lower()] for letter in word]
    for i in range(0, 16 - len(indexarray)):
        indexarray.append(0)
    return indexarray

def generateOutput(languageCode):
    outputarray = [0] * 4

    outputarray[outputContext[languageCode]] = 1

    return outputarray


def main():

    for i in range(len(letters)):
        letterDict[letters[i]] = i + 1

    with open(f"/home/mattecatte/STEM/ml/new/data/scripts/dump/datafile.json", "r") as jsonfile:
        json = j.load(jsonfile)

        for language, words in json.items():
            print(f"Processing language: '{language}'")
            for word in words:
                grandArray.append({
                    "output" : generateOutput(language),
                    "input" : index(word),
                    "human symbols" : word,
                    "language" : language
                    })


    for key in outputContext.keys():
        outputContext[key] = generateOutput(key)

    random.shuffle(grandArray)

    for i in range(100):
        for element in grandArray:
            outputArray.append(element["output"])
            inputArray.append(element["input"])
        
    print(f"\n~Processed {len(grandArray)} elements~")

    return np.array(inputArray, dtype=float), np.array(outputArray, dtype=float)

