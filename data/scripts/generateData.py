import requests
import json
import unidecode

url = "https://translate.yandex.net/api/v1.5/tr.json/translate"

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

rqParams = {
    "key" : "trnsl.1.1.20190604T204140Z.6a1166565ba3946d.6e89f6f5ba4aab6f5c1b1d76876a75394b2a6a91"
}

data = {}

with open("dump/initial.txt", "r") as datafile:

    print("Reading initialdata file")
    filetext = datafile.read().split("\n")
    
    data["en"] = filetext
    
    
    
    for words in [filetext[0:1000], filetext[1001: 2001], filetext[2002:]]:

        rqParams["text"] = words

        for language in params["languages"][1:]:
            
            print(f"Requesting for language: '{language}'")
        
            rqParams["lang"] = language
        
            r = requests.get(url, rqParams)
        
            r.raise_for_status()
        
            jsonData = r.json()
        
            text = jsonData["text"]
        
            data[language] = text

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

with open("dump/datafile.json", "w+") as dump:
    print("Dumping data...")
    json.dump(data, dump)

        
        
    

        