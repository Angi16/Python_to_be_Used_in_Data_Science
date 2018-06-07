**Dependencies:-**

Install all the dependencies using pip:-

    Scikit-learn (http://scikit-learn.org/stable/install.html)
    numpy (pip install numpy)
    scipy (pip install scipy)

**Overview**

This code classifies gender.
On a small dataset of body metrics (height, width, and shoe size) labeled male or female,  the code uses the scikit-learn machine learning library to train following 4 classifier algorithms:-

1. Decision Tree
2. Support Vector Machine
3. Perceptron
4. K-Nearest-Neighbours

Then train them on the same dataset and compare their results.

We can determine accuracy by trying to predict testing our trained classifier on samples from the training data and see if it correctly classifies it.

**Usage**

Once you have your dependencies installed via pip, run the script in terminal via-

python GenClf.py
