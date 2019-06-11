import json
import unidecode

arrays = {
    "fr" : [],
    "de" : [],
    "en" : [],
    "la" : []
}

with open("fr.txt", "r") as scrubTarget:
    for line in scrubTarget:
        arrays["fr"].append(unidecode.unidecode(line.partition(" ")[0]))

with open("de.txt", "r") as scrubTarget:
    for line in scrubTarget:
        arrays["de"].append(unidecode.unidecode(line.partition(" ")[0]))

with open("initial_en", "r") as scrubTarget:
    arrays["en"] = scrubTarget.read().split("\n")

with open("initial_la", "r") as scrubTarget:
    for line in scrubTarget:
        word_array = line.partition(" : ")[0].split(" ")

        for word in word_array:
            word = unidecode.unidecode(word)

        lenarray = [len(word) for word in word_array]

        longestword = word_array[
            lenarray.index(max(lenarray))
        ]

        arrays["la"].append(longestword)

for key, value in arrays.items():
    word_i = 0
    for word_i in range(len(value)):
        word = value[word_i][:16]
        # No idea where any of these symbols came from but they have no place
        value[word_i] = unidecode.unidecode(word.lower()
        .replace("^", "")
        .replace(" ", "")
        .replace("-", "")
        .replace(":", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("=", "")
        .replace(".", "")
        .replace("+", "").strip())
        word_i += 1

with open("datafile.json", "w+") as landfill:
    json.dump(arrays, landfill)