
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import operator
import numpy as np
class method:
    # make stopword list
    stop_words = set(stopwords.words("english"))

    #method for crating vector corresponding to each question
    def word_vectorization(self,wvec, question):
        vector_value = []
        for word in question:
            addvalues = [0] * 50
            words = word_tokenize(word)
            for w in words:
                #convert to lower
                w = w.lower()
                #check word is not stopword
                if w not in self.stop_words:
                    try:
                        addvalues = map(operator.add, wvec[w], addvalues)
                    except KeyError:
                        continue
            row = [vec for vec in addvalues]
            row = np.sum(row)
            if row:
                if vector_value:
                    np.append(vector_value, row)
                else:
                    vector_value = row
        return vector_value
    def data_reading(self,filename):
        file = open(filename, 'r')

        labeled_data = []
        for line in file:
            sentence, category = line.strip().split(' ,,, ')
            labeled_data.append((sentence, category))
        return labeled_data