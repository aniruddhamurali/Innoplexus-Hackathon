from skmultilearn.adapt import MLkNN
import scipy
from scipy.io import arff

from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

data, meta = scipy.io.arff.loadarff('/Users/aniruddha/Desktop/Innoplexus-Hackathon/yeast/yeast-train.arff')
#df = pd.DataFrame(data)

from sklearn.datasets import make_multilabel_classification

# this will generate a random multi-label dataset
X, y = make_multilabel_classification(sparse = True, n_labels = 20,
return_indicator = 'sparse', allow_unlabeled = False)


'''trainName = 'information_train.csv'
testName = 'information_test.csv'
train = "train.csv"
# Row 1: Abstract (text)
# Row 2: Article title
# Row 3: Author
# Row 4: pmid
# Row 5: Publication Date
# Row 6: Set

X_train = []
test_data = []
train_results = []

with open(trainName, 'rb') as file:
    X_train = [line.decode('utf-8').strip().split('\t') for line in file.readlines()]
with open(testName, 'rb') as file:
    test_data = [line.decode('utf-8').strip().split('\t') for line in file.readlines()]

X_train = [X_train[index] for index in range(0,len(X_train)) if X_train[index] != ['"']]
X_train = [X_train[index] for index in range(0,len(X_train)) if X_train[index] != ['']]
test_data = [test_data[index] for index in range(0,len(test_data)) if test_data[index] != ['"']]
test_data = [test_data[index] for index in range(0,len(test_data)) if test_data[index] != ['']]

with open(train, 'r') as file:
    for line in file.readlines():
        index = line.index(',')'''

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
classifier.fit(X, y)

# predict
predictions = classifier.predict(data)

accuracy_score(meta,predictions)
        
    


'''classifier = MLkNN(k=20)

# train
classifier.fit(data, meta)

# predict
predictions = classifier.predict(X)

accuracy_score(Y,predictions)'''
