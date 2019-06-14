letters = "abcdefghijklmnopqrstuvwxyz"

# Converts letters into numbers
letterDict = {}

outputContext = {
    0 : "English",
    1 : "French",
    2 : "German",
    3 : "Latin"
}

for i in range(len(letters)):
    letterDict[letters[i]] = i + 1

def index(word):

    indexarray = []

    for letter in word:
        indexarray.append(
            [letterDict[letter.lower()]]
            )
    for i in range(0, 16 - len(indexarray)):
        indexarray.append([0])

    print(f"\n  Input array: {indexarray}\n")

    return indexarray

def contextualize(networkOutput):
    return outputContext[networkOutput[0]]

def validate(word):
    badchar = None

    try:
        for letter in word:
            if letter not in letters:
                badchar = letter
                raise Exception

        return True, None
            
    except:
        return False, badchar