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


# Return difference between years of publication
def yearDiff(y):
    return 2018 - y

    

testName = 'information_test.csv'
test_data = []

with open(testName, 'rb') as file:
    test_data = [line.decode('utf-8').strip().split('\t') for line in file.readlines()]

test_data = [test_data[index] for index in range(0,len(test_data)) if test_data[index] != ['"']]
test_data = [test_data[index] for index in range(0,len(test_data)) if test_data[index] != ['']]

raw_documents1 = [test_data[index][0] for index in range(0,len(test_data))] # Abstract
raw_documents2 = [test_data[index][1] for index in range(0,len(test_data))] # Title
raw_documents3 = [test_data[index][2] for index in range(0,len(test_data))] # Author
raw_documents4 = [test_data[index][0] + ' ' + test_data[index][1] + ' ' + test_data[index][2] for index in range(0,len(test_data))] # Abstract


# Generate sims for abstracts
gen_docs1 = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents1]
dictionary1 = gensim.corpora.Dictionary(gen_docs1)
corpus1 = [dictionary1.doc2bow(gen_doc) for gen_doc in gen_docs1]
tf_idf1 = gensim.models.TfidfModel(corpus1)
sims1 = gensim.similarities.Similarity('/Users/aniruddha/Desktop/Innoplexus-Hackathon',tf_idf1[corpus1],
                                      num_features=len(dictionary1))

# Generate sims for titles
gen_docs2 = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents2]
dictionary2 = gensim.corpora.Dictionary(gen_docs2)
corpus2 = [dictionary2.doc2bow(gen_doc) for gen_doc in gen_docs2]
tf_idf2 = gensim.models.TfidfModel(corpus2)
sims2 = gensim.similarities.Similarity('/Users/aniruddha/Desktop/Innoplexus-Hackathon',tf_idf2[corpus2],
                                      num_features=len(dictionary2))

# Generate sims for authors
gen_docs3 = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents3]
dictionary3 = gensim.corpora.Dictionary(gen_docs3)
corpus3 = [dictionary3.doc2bow(gen_doc) for gen_doc in gen_docs3]
tf_idf3 = gensim.models.TfidfModel(corpus3)
sims3 = gensim.similarities.Similarity('/Users/aniruddha/Desktop/Innoplexus-Hackathon',tf_idf3[corpus3],
                                      num_features=len(dictionary3))

# Generate sims for authors
gen_docs4 = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents4]
dictionary4 = gensim.corpora.Dictionary(gen_docs4)
corpus4 = [dictionary4.doc2bow(gen_doc) for gen_doc in gen_docs4]
tf_idf4 = gensim.models.TfidfModel(corpus4)
sims4 = gensim.similarities.Similarity('/Users/aniruddha/Desktop/Innoplexus-Hackathon',tf_idf4[corpus4],
                                      num_features=len(dictionary4))


resultsDict = dict()


def addLists(l1, l2, l3, l4):
    result = []
    for i in range(0,len(l1)):
        #result.append(l1[i] + l2[i] + l3[i])
        result.append(l1[i] + l2[i] + l3[i] + l4[i])
        if i == 0:
            continue
        year = int(test_data[i][4][0:4])
        if yearDiff(year) < 5:
            result[i] *= 1.1
        elif yearDiff(year) < 10:
            result[i] *= 1.05
    return result

def multiplyList(array, weightage):
    result = []
    for i in range(0,len(array)):
        result.append(array[i] * weightage)
    return result



def returnResults(rawdocs1, rawdocs2, rawdocs3, rawdocs4):
    # Similarities between abstracts
    query_doc1 = [w.lower() for w in word_tokenize(raw_documents1[i])]
    query_doc_bow1 = dictionary1.doc2bow(query_doc1)
    query_doc_tf_idf1 = tf_idf1[query_doc_bow1]
    results1 = sims1[query_doc_tf_idf1]
    
    # Similarities between titles
    query_doc2 = [w.lower() for w in word_tokenize(raw_documents2[i])]
    query_doc_bow2 = dictionary2.doc2bow(query_doc2)
    query_doc_tf_idf2 = tf_idf2[query_doc_bow2]
    results2 = sims2[query_doc_tf_idf2]
    
    # Similarities between authors
    query_doc3 = [w.lower() for w in word_tokenize(raw_documents3[i])]
    query_doc_bow3 = dictionary3.doc2bow(query_doc3)
    query_doc_tf_idf3 = tf_idf3[query_doc_bow3]
    results3 = sims3[query_doc_tf_idf3]

    # Similarities between abstract+title
    query_doc4 = [w.lower() for w in word_tokenize(raw_documents4[i])]
    query_doc_bow4 = dictionary4.doc2bow(query_doc4)
    query_doc_tf_idf4 = tf_idf4[query_doc_bow4]
    results4 = sims4[query_doc_tf_idf4]

    # Weightages
    '''w1 = 1/3
    w2 = 1/3
    w3 = 1/3'''
    w1 = 1/4
    w2 = 1/4
    w3 = 1/4
    w4 = 1/4

    # Modified results based on weightages
    results1 = multiplyList(results1, w1)
    results2 = multiplyList(results2, w2)
    results3 = multiplyList(results3, w3)
    results4 = multiplyList(results4, w4)
    results = addLists(results1, results2, results3, results4)

    return results


for i in range(0,len(test_data)):
    results = returnResults(raw_documents1, raw_documents2, raw_documents3, raw_documents4)
    temp = []
    tempMax = 0
    maxJ = 0

    for j in range(0, len(test_data)):
        if i == j:
            continue
        if results[j] > .175 and test_data[i][5] == test_data[j][5] and dateNewer(test_data[i][4], test_data[j][4]) == True:
            temp.append(test_data[j][3])
        else:
            if results[j] > tempMax and test_data[i][5] == test_data[j][5]:
                tempMax = results[j]
                maxJ = j

    if len(temp) == 0:
        resultsDict[test_data[i][3]] = str([test_data[maxJ][3]])
    else:
        resultsDict[test_data[i][3]] = str(temp)


del resultsDict['pmid']
keys = list(resultsDict.keys())
values = list(resultsDict.values())

open('solution3.csv', 'w').close()
with open('solution3.csv', 'w') as file:
    file.write('pmid,ref_list\n')
    for i in range(0,len(keys)):
        if values[i].count(',') == 0:
            file.write(keys[i] + ',' + values[i] + "\n")
        else:
            file.write(keys[i] + ',"' + values[i] + '"' + "\n")
            
        

    






