# README #


Identify Question Type: Given a question, the aim is to identify the category it belongs to.
The four categories to handle for this assignment are : Who, What, When, Affirmation(yes/no).
Label any sentence that does not fall in any of the above four as "Unknown" type.


### What is this repository? ###
In this I have created a model to identify the type of question as Who, What, When, Affirmation(yes/no)
and unknown. Steps followed in the programme :
1- Read data fron given labelleddata.txt and segregate sentence and label

2- Convert question sentence to corresponding vector using glove.6B.50d.txt

3- Created multinomial logistic regression classification model for the dataset

4- Validate our model with all dataset predicted value

I choose multinomial logistic regression model because it is a classification
method that generalizes logistic regression to multiclass problems,
i.e. with more than two possible discrete outcomes.
That is, it is a model that is used to predict the probabilities of
the different possible outcomes of a categorically distributed dependent
variable, given a set of independent variables.It is best suited to our case.


###Dataset Used####
* glove.6B.50d.txt  dataset is required

+ Dataset is available on http://nlp.stanford.edu/projects/glove/

+ Download it and put in the code folder for proper execution.

#### Reqirement #####
+ Package
     * 1.Gensim
     * 2.Numpy
     * 3.Sklearn
     * 4.Tensorflow

#### Run Code #####
* python main.py


### Contributor ###
Vijay Anand

201452002@iiitvadodara.ac.in
