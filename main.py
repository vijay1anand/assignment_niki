
import random
from sklearn import linear_model
import gensim
import numpy as np
from method import method

#checking accuracy of our prediction
def accuracy(predicted, actuals):
    success = [np.any(val[0] == val[1]) for val in zip(predicted, actuals)]
    return (success.count(True) / len(success)) * 100


object = method()

#reading labelled data
labelled_data = object.data_reading('dataset/LabelledData (1).txt')

random.shuffle(labelled_data)



# load  predefined vector model
word2vector = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.50d.txt', binary=False)

# question list
question_data = [element[0] for element in labelled_data]

# making equivalent vectors corresponding to question
print("Word to vector creation")
question_list = [object.word_vectorization(word2vector, [element[0]]) for element in labelled_data]

# type of labels for question
question_label = [labels[1] for labels in labelled_data]

# creating multinomial classifier model for the dataset
print("Model Creating")
model_classifier = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
#fit model with initial 900 data
model_classifier.fit(np.array(question_list[:900]).reshape(-1, 1), np.array(question_label[:900]))

#predict classification for trained value of question using model
pred_train = [model_classifier.predict(np.array(question_list[:900]).reshape(-1, 1))]

#predict classification for all value of question using model
pred_test = [model_classifier.predict(np.array(question_list).reshape(-1, 1))]

print ("Checking accuracy")
print ("Train accuracy " + str(accuracy(pred_train, question_label[:900])))
print ("Test accuracy " + str(accuracy(pred_test, question_label)))
