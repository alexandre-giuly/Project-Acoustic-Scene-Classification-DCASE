import random
import unicodedata
import string
import numpy as np
import sentencepiece as spm
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


all_letters = string.ascii_letters + " .,;'-?!_"

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename,pos):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    for i in range(len(lines)):
        lines[i] = lines[i].split()[pos]
        
    return lines

def writeLines(filename, lines):
    file = open(filename, 'w')
    for line in lines:
        file.write(line)
        file.write('\n')


n_labels = 10
n_big_labels = 3

labelind2name = {
        0: "airport",
        1: "bus",
        2: "metro",
        3: "metro_station",
        4: "park",
        5: "public_square",
        6: "shopping_mall",
        7: "street_pedestrian",
        8: "street_traffic",
        9: "tram",
    }
name2labelind = {
        "airport": 0,
        "bus": 1,
        "metro": 2,
        "metro_station": 3,
        "park": 4,
        "public_square": 5,
        "shopping_mall": 6,
        "street_pedestrian": 7,
        "street_traffic": 8,
        "tram": 9,
    }

labelind2name2 = {
        0: "indoor",
        1: "outdoor",
        2: "transportation",
    }
name2labelind2 = {
        "airport": 0,
        "bus": 2,
        "metro": 2,
        "metro_station": 0,
        "park": 1,
        "public_square": 1,
        "shopping_mall": 0,
        "street_pedestrian": 1,
        "street_traffic": 1,
        "tram": 2,
    }

def confus_matrix(filename1, filename2,n):

    
    confusion = torch.zeros(n, n)

    lines1 = readLines(filename1,1)[1:]
    lines2 = readLines(filename2,1)[1:]
    
    if n==n_labels:
        lbl = name2labelind
        all_categories = list(lbl.keys())
    
    if n==n_big_labels:
        lbl = name2labelind2
        all_categories = list(labelind2name2.values())
    # Go through a bunch of examples and record which are correctly guessed
    for i in range(len(lines1)):
        guess_i = lbl[lines1[i]]
        category_i = lbl[lines2[i]]
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    
    
    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

confus_matrix("./evaluation_setup/resultkd.csv","./evaluation_setup/fold1_evaluate.csv",n_big_labels)


'''lines2 = readLines("./evaluation_setup/fold1_test.csv",0)
writeLines("./evaluation_setup/fold1_test.csv",lines2)



lines1 = readLines("./evaluation_setup/fold1_evaluate.csv",0)
lines2 = readLines("./evaluation_setup/fold1_test.csv",0)
lines3 = readLines("./evaluation_setup/fold1_train.csv",0)




s=0
for filename in os.listdir("audio"):
    #print(filename)
    var = False
    
    filename2 = "audio/"+filename
    #print(filename2)
    if filename2 in lines1:
        var = True
        
    if filename2 in lines2:
        var = True
        
    if filename2  in lines3:
        var = True

    if not var:
       os.remove("./audio/"+ filename)'''



       




    
#if not var:
    #    os.remove("./audio/"+ filename)
