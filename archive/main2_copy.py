import gensim

'''raw_documents = ["I'm taking the show on the road.",
                 "My socks are a force multiplier.",
             "I am the barber who cuts everyone's hair who doesn't cut their own.",
             "Legend has it that the mind is a mad monkey.",
            "I make my own fun."]

from nltk.tokenize import word_tokenize
gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents]

dictionary = gensim.corpora.Dictionary(gen_docs)

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

tf_idf = gensim.models.TfidfModel(corpus)

sims = gensim.similarities.Similarity('/Users/aniruddha/Desktop/Innoplexus-Hackathon',tf_idf[corpus],
                                      num_features=len(dictionary))

query_doc = [w.lower() for w in word_tokenize("Socks are a force for good.")]
print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)

print(sims[query_doc_tf_idf])'''


testName = 'information_test.csv'
test_data = []
with open(testName, 'rb') as file:
    test_data = [line.decode('utf-8').strip().split('\t') for line in file.readlines()]
    #print(rows[2])
#raw_documents = [row[0] for row in test_data if row[0] != "abstract"]
    
'''raw_documents = []
index = 0
while len(raw_documents) < 2000:
    raw_documents.append(test_data[index][0])
    index += 1'''

raw_documents = [test_data[index][0] for index in range(0,len(test_data))]


from nltk.tokenize import word_tokenize
gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents]

dictionary = gensim.corpora.Dictionary(gen_docs)

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

tf_idf = gensim.models.TfidfModel(corpus)

sims = gensim.similarities.Similarity('/Users/aniruddha/Desktop/Innoplexus-Hackathon',tf_idf[corpus],
                                      num_features=len(dictionary))

'''query_doc = [w.lower() for w in word_tokenize(raw_documents[1])]
#print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
#print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
#print(query_doc_tf_idf)

results = sims[query_doc_tf_idf]'''
'''maximum = 0
for i in results:
    if i > maximum and i < .95:
        maximum = i
print(maximum)
#print(sims[query_doc_tf_idf])'''

resultsDict = dict()

for i in range(0,5):
    query_doc = [w.lower() for w in word_tokenize(raw_documents[i])]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    results = sims[query_doc_tf_idf]
    temp = []
    for j in range(i+1, len(test_data)):
        if results[j] > .2:
            temp.append(test_data[j][3])
    resultsDict[test_data[i][3]] = str(temp)

print(resultsDict)
            
        

    






