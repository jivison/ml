# For the yandex API
import requests

# For writing the raw data
import json

# To get rid of accents etc.
import unidecode

# API
url = "https://translate.yandex.net/api/v1.5/tr.json/translate"

# Which languages it will translate to (review the documentation for the codes)
params = {
    "languages" : [
     "en",
     "fr", 
     "de", 
     "la", 
    #  "mi", 
    #  "es", 
    #  "sw", 
    #  "et"
     ],
}

# The params passed to the API
rqParams = {}

# Read the apikey
with open("apikey", "r") as apikey:
    rqParams["key"] = apikey.read()

# Holds the data fetched by the api
data = {}

# Open the words that will be translated
with open("dump/initial.txt", "r") as datafile:

    print("Reading initialdata file")
    
    # Make a big array of all the words
    filetext = datafile.read().split("\n")
    
    # No need to translate english to english
    data["en"] = filetext
    
    # The api has a maxiumum param count, so splits the datafile into three parts of about 1000
    for words in [filetext[0:1000], filetext[1001: 2001], filetext[2002:]]:

        # Set the requests params
        rqParams["text"] = words

        # Translate for each language (except English)
        for language in params["languages"][1:]:
            
            print(f"Requesting for language: '{language}'")
        
            rqParams["lang"] = language

            # Call the API
            r = requests.get(url, rqParams)
        
            # Raise an error for a bad status code
            r.raise_for_status()

            # Read the json
            jsonData = r.json()

            text = jsonData["text"]
        
            data[language] = text

# Scrubs the data
for language, wordarray in data.items():

    print(f"Formatting words for language: '{language}'")

    copyArray = wordarray.copy()

    # I am so sorry
    wordarray = [
        unidecode.unidecode(word).partition(" ")[2].strip().replace("l'", "").replace(" ", "").replace("-", "").replace("'", "").replace(".", "").replace(",", "").replace("?", "").replace(":", "")[:16]
        if unidecode.unidecode(word).partition(" ")[0] in ["la", "le", "les", "en", "au", "de", "ne", "se", "a", "e", "i", "te", "los", "el"] 
        and unidecode.unidecode(word).partition(" ")[2] != ""
        else unidecode.unidecode(word).partition(" ")[0].replace(" ", "").replace("-", "").replace("l'", "").replace("'", "").replace(".", "").replace(",", "").replace("?", "").replace(":", "")[:16]
        for word in copyArray
        ]

    data[language] = wordarray

# Dumps the data
with open("dump/datafile.json", "w+") as dump:
    print("Dumping data...")
    json.dump(data, dump)

        
        
    

        