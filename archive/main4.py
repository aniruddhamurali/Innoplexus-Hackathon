import gensim
from nltk.tokenize import word_tokenize


# Checks if d1 is more recent than d2
def dateNewer(d1, d2):
    y1 = int(d1[0:4])
    m1 = int(d1[5:7])
    a1 = int(d1[8:10])
    y2 = int(d2[0:4])
    m2 = int(d2[5:7])
    a2 = int(d2[8:10])

    if y1 > y2:
        return True
    else:
        if m1 > m2:
            return True
        elif a1 > a2:
            return True
    return False


testName = 'information_test.csv'
test_data = []

with open(testName, 'rb') as file:
    test_data = [line.decode('utf-8').strip().split('\t') for line in file.readlines()]

test_data = [test_data[index] for index in range(0,len(test_data)) if test_data[index] != ['"']]

test_data = [test_data[index] for index in range(0,len(test_data)) if test_data[index] != ['']]

raw_documents = [test_data[index][0] + ' ' + test_data[index][1] + ' ' + test_data[index][2] for index in range(0,len(test_data))]
gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents]

dictionary = gensim.corpora.Dictionary(gen_docs)
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]


tf_idf = gensim.models.TfidfModel(corpus)


sims = gensim.similarities.Similarity('/Users/aniruddha/Desktop/Innoplexus-Hackathon',tf_idf[corpus],
                                      num_features=len(dictionary))
resultsDict = dict()



for i in range(0,len(test_data)):
    query_doc = [w.lower() for w in word_tokenize(raw_documents[i])]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    results = sims[query_doc_tf_idf]
    temp = []
    tempMax = 0
    maxJ = 0

    for j in range(0, len(test_data)):
        if i == j:
            continue
        if results[j] > .175 and test_data[i][5] == test_data[j][5] and dateNewer(test_data[i][4], test_data[j][4]) == True:
            temp.append(test_data[j][3])
        else:
            if results[j] > tempMax:
                tempMax = results[j]
                maxJ = j
                
    if len(temp) == 0:
        resultsDict[test_data[i][3]] = str([test_data[maxJ][3]])
    else:
        resultsDict[test_data[i][3]] = str(temp)


del resultsDict['pmid']
keys = list(resultsDict.keys())
values = list(resultsDict.values())

open('solution2.csv', 'w').close()
with open('solution2.csv', 'w') as file:
    file.write('pmid,ref_list\n')
    for i in range(0,len(keys)):
        if values[i].count(',') == 0:
            file.write(keys[i] + ',' + values[i] + "\n")
        else:
            file.write(keys[i] + ',"' + values[i] + '"' + "\n")
            
        

    






