# Soal 2 - Rekomendasi Buku Bagus
# ================================

import numpy as np
import pandas as pd

# read the csv files
dataBook = pd.read_csv(
    'books.csv'
)

dataRating = pd.read_csv(
    'ratings.csv'
)

# add a new col: 'authors' + 'original_title' + 'title' + 'language_code'
def mergeCol(i):
    return str(i['authors']) + ' ' + str(i['original_title']) + ' ' + str(i['title']) + ' ' + str(i['language_code'])

dataBook['Attribute'] = dataBook.apply(mergeCol, axis='columns')
# print(dataBook)

# =============================================================================================
# count vectorizer

from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(
    tokenizer = lambda x: x.split(' ')
)
matrixAttribute = model.fit_transform(dataBook['Attribute'])

attribute = model.get_feature_names()
jumlahAttribute = len(attribute)

# print(attribute)
# print(jumlahAttribute)

# =============================================================================================
# cosine similarity

from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(matrixAttribute)
# print(score)

# =============================================================================================
# testing
# only books that have ratings of 3 stars and higher will be included in each's favourite book

# based on the given profile, Andi has 4 books with 3 stars and above
andiFirst = dataBook[dataBook['original_title'] == 'The Hunger Games']['book_id'].tolist()[0]-1
andiSecond = dataBook[dataBook['original_title'] == 'Catching Fire']['book_id'].tolist()[0]-1
andiThird = dataBook[dataBook['original_title'] == 'Mockingjay']['book_id'].tolist()[0]-1
andiFourth = dataBook[dataBook['original_title'] == 'The Hobbit or There and Back Again']['book_id'].tolist()[0]-1
andiFavourite = [andiFirst, andiSecond, andiThird, andiFourth]

# Budi has 3 books with 3 stars and above
budiFirst = dataBook[dataBook['original_title'] == "Harry Potter and the Philosopher's Stone"]['book_id'].tolist()[0]-1
budiSecond = dataBook[dataBook['original_title'] == 'Harry Potter and the Chamber of Secrets']['book_id'].tolist()[0]-1
budiThird = dataBook[dataBook['original_title'] == 'Harry Potter and the Prisoner of Azkaban']['book_id'].tolist()[0]-1
budiFavourite = [budiFirst, budiSecond, budiThird]

# Ciko has 1 book with 3 stars and above
cikoFirst = dataBook[dataBook['original_title'] == 'Robots and Empire']['book_id'].tolist()[0]-1
cikoFavourite = [cikoFirst]

# Dedi has 3 books with 3 stars and above
dediFirst = dataBook[dataBook['original_title'] == 'Nine Parts of Desire: The Hidden World of Islamic Women']['book_id'].tolist()[0]-1
dediSecond = dataBook[dataBook['original_title'] == 'A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam']['book_id'].tolist()[0]-1
dediThird = dataBook[dataBook['original_title'] == 'No god but God: The Origins, Evolution, and Future of Islam']['book_id'].tolist()[0]-1
dediFavourite = [dediFirst, dediSecond, dediThird]

# Ello has 3 books with 3 stars and above
elloFirst = dataBook[dataBook['original_title'] == 'Doctor Sleep']['book_id'].tolist()[0]-1
elloSecond = dataBook[dataBook['original_title'] == 'The Story of Doctor Dolittle']['book_id'].tolist()[0]-1
elloThird = dataBook[dataBook['title'] == "Bridget Jones's Diary (Bridget Jones, #1)"]['book_id'].tolist()[0]-1
elloFavourite = [elloFirst, elloSecond, elloThird]

# =============================================================================================
# creating score list for each member

andiFirst_scoreList = list(enumerate(score[andiFirst]))
andiSecond_scoreList = list(enumerate(score[andiSecond]))
andiThird_scoreList = list(enumerate(score[andiThird]))
andiFourth_scoreList = list(enumerate(score[andiFourth]))

budiFirst_scoreList = list(enumerate(score[budiFirst]))
budiSecond_scoreList = list(enumerate(score[budiSecond]))
budiThird_scoreList = list(enumerate(score[budiThird]))

ciko_scoreList = list(enumerate(score[cikoFirst]))

dediFirst_scoreList = list(enumerate(score[dediFirst]))
dediSecond_scoreList = list(enumerate(score[dediSecond]))
dediThird_scoreList = list(enumerate(score[dediThird]))

elloFirst_scoreList = list(enumerate(score[elloFirst]))
elloSecond_scoreList = list(enumerate(score[elloSecond]))
elloThird_scoreList = list(enumerate(score[elloThird]))

# =============================================================================================

andi_scoreList = []
for i in andiFirst_scoreList:
    andi_scoreList.append((i[0], 0.25 * (andiFirst_scoreList[i[0]][1] + andiSecond_scoreList[i[0]][1] + andiThird_scoreList[i[0]][1] + andiFourth_scoreList[i[0]][1])))

budi_scoreList = []
for i in andiFirst_scoreList:
    budi_scoreList.append((i[0], (budiFirst_scoreList[i[0]][1] + budiSecond_scoreList[i[0]][1] + budiThird_scoreList[i[0]][1])/3))

dedi_scoreList = []
for i in andiFirst_scoreList:
    dedi_scoreList.append((i[0], (dediFirst_scoreList[i[0]][1] + dediSecond_scoreList[i[0]][1] + dediThird_scoreList[i[0]][1])/3))

ello_scoreList = []
for i in andiFirst_scoreList:
    ello_scoreList.append((i[0], (elloFirst_scoreList[i[0]][1] + elloSecond_scoreList[i[0]][1] + elloThird_scoreList[i[0]][1])/3))


sortAndi = sorted(
    andi_scoreList,
    key = lambda j: j[1],
    reverse = True
)

sortBudi = sorted(
    budi_scoreList,
    key = lambda j: j[1],
    reverse = True
)

sortCiko = sorted(
    ciko_scoreList,
    key = lambda j: j[1],
    reverse = True
)

sortDedi = sorted(
    dedi_scoreList,
    key = lambda j: j[1],
    reverse = True
)

sortEllo = sorted(
    ello_scoreList,
    key = lambda j: j[1],
    reverse = True
)

# =============================================================================================
# show top 5 recommendations for each profile

andiSimilar = []
for i in sortAndi:
    if i[1] > 0:
        andiSimilar.append(i)

budiSimilar = []
for i in sortBudi:
    if i[1] > 0:
        budiSimilar.append(i)

cikoSimilar = []
for i in sortCiko:
    if i[1] > 0:
        cikoSimilar.append(i)

dediSimilar = []
for i in sortDedi:
    if i[1] > 0:
        dediSimilar.append(i)

elloSimilar = []
for i in sortEllo:
    if i[1] > 0:
        elloSimilar.append(i)


print('1. Buku bagus untuk Andi: ')
for i in range(0, 5):
    if andiSimilar[i][0] not in andiFavourite:
        print('-', dataBook['original_title'].iloc[andiSimilar[i][0]])
    else:
        i += 5
        print('-', dataBook['original_title'].iloc[andiSimilar[i][0]])

print('\n2. Buku bagus untuk Budi: ')
for i in range(0, 5):
    if budiSimilar[i][0] not in budiFavourite:
        print('-', dataBook['original_title'].iloc[budiSimilar[i][0]])
    else:
        i += 5
        print('-', dataBook['original_title'].iloc[budiSimilar[i][0]])

print('\n3. Buku bagus untuk Ciko: ')
for i in range(0, 5):
    if cikoSimilar[i][0] not in cikoFavourite:
        print('-', dataBook['original_title'].iloc[cikoSimilar[i][0]])
    else:
        i += 5
        print('-', dataBook['original_title'].iloc[cikoSimilar[i][0]])

print('\n4. Buku bagus untuk Dedi: ')
for i in range(0, 5):
    if dediSimilar[i][0] not in dediFavourite:
        print('-', dataBook['original_title'].iloc[dediSimilar[i][0]])
    else:
        i += 5
        print('-', dataBook['original_title'].iloc[dediSimilar[i][0]])

print('\n5. Buku bagus untuk Ello: ')
for i in range(0, 5):
    if elloSimilar[i][0] not in elloFavourite:
        if str(dataBook['original_title'].iloc[elloSimilar[i][0]]) == 'nan':
            print('-', dataBook['title'].iloc[elloSimilar[i][0]])
        else:
            print('-', dataBook['original_title'].iloc[elloSimilar[i][0]])
    else:
        i += 5
        if str(dataBook['original_title'].iloc[elloSimilar[i][0]]) == 'nan':
            print('-', dataBook['title'].iloc[elloSimilar[i][0]])
        else:
            print('-', dataBook['original_title'].iloc[elloSimilar[i][0]])