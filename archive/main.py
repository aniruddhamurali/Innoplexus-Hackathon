import csv
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

trainName = 'information_train.csv'
testName = 'information_test.csv'
# Row 1: Abstract (text)
# Row 2: Article title
# Row 3: Author
# Row 4: pmid
# Row 5: Publication Date
# Row 6: Set

train_data = []
test_data = []
with open(trainName, 'rb') as file:
    train_data = [line.decode('utf-8').strip().split('\t') for line in file.readlines()]
with open(testName, 'rb') as file:
    test_data = [line.decode('utf-8').strip().split('\t') for line in file.readlines()]

test_data = [test_data[index] for index in range(0,len(test_data)) if test_data[index] != ['"']]
test_data = [test_data[index] for index in range(0,len(test_data)) if test_data[index] != ['']]
#print(len(test_data))

# Define test data (text)
input_data = [test_data[index][0] for index in range(0,len(test_data))]


    
# Define the category map
category_map = {'talk.politics.misc': 'Politics', 'rec.autos': 'Autos',
                'rec.sport.hockey': 'Hockey', 'sci.electronics': 'Electronics',
                'sci.med': 'Medicine'}

#training set
training_data = fetch_20newsgroups(subset='train',
                                   categories=category_map.keys(),shuffle=True,
                                   random_state=5)


# Build a count vectorizer and extract term counts
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(training_data)

# Create the tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)



# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB().fit(train_tfidf, training_data.target)

# Transform input data using count vectorizer
input_tc = count_vectorizer.transform(input_data)

# Transform vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)

# Predict the output categories
predictions = classifier.predict(input_tfidf)

# Print the outputs
count = 0
for sent, category in zip(input_data, predictions):
    print('\nInput:', sent, '\nPredicted category:', \
          category_map[training_data.target_names[category]])
    count += 1
    if count == 5:
        break

