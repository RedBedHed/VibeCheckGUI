import numpy as np
import re
import math
import pickle
import tkinter as tk
import tkinter.font as tf

# From https://www.geeksforgeeks.org/removing-stop-words-nltk-python/#, adapted for this program.
stopwords = set(
    ['this', 'dont', 'yours', 'his', 'can', 'weren', 'themselves', 'hasnt', 'do', 'at', 'during', 
     'their', 'them', 'mightnt', 'were', 'wouldn', 'haven', 'has', 'couldn', 'myself', 'that', 
     'wasn', 'neednt', "youre", 'while', 'it', 'nor', 'm', 'doesnt', 'just', 'himself', 'with', 
     'youd', 'until', 'a', 'does', 'where', 'shes', 've', 'arent', 'ours', 'under', 'ourselves', 
     'are', 'you', 'be', 'once', 'aren', 'having', 'on', 'to', 'below', 'not', 'such', 'itself', 
     'but', 're', 'of', 'yourselves', 'then', 'ain', 'thatll', 'isnt', 'being', 'same', 'through', 
     'further', 'up', 'how', 'doesn', 'her', 'very', 'couldnt', 'werent', 'which', 'he', 'me', 'in', 
     'each', 'we', 'havent', 'isn', 'doing', 'because', 's', 'hasn', 'shant', 'down', 'as', 'didnt', 
     'only', 'herself', 'before', 'don', 'and', 'against', 'what', 'by', 'wont', 'for', 'so', 'above', 
     'been', 'or', 'again', 'shouldnt', 'whom', 'why', 'here', 'shouldve', 'hadn', 'shan', 'those', 'o', 
     'is', 'about', 'over', 'there', 't', 'after', 'youll', 'the', 'who', 'did', "you've", 'mustn', 'too', 
     'mightn', 'll', 'hadnt', 'if', 'was', 'both', 'am', 'these', 'most', 'they', 'few', 'off', 'now', 'had', 
     'out', 'our', 'into', 'needn', 'wasnt', 'between', 'yourself', 'she', 'other', 'an', 'shouldn', 'y', 'some', 
     'hers', 'i', 'ma', 'him', 'when', 'will', 'all', 'own', 'any', 'than', 'have', 'wouldnt', 'didn', 'mustnt', 
     'my', 'no', 'theirs', 'd', 'your', 'from', 'its', 'won', 'should', 'more']
)

# Class labels 1-6
CLASS_LABELS = ['sadness','joy','love','anger','fear','surprise']
CLASS_EMOJIS = ['\U0001F641', '\U0001F642','\U0001F60D','\U0001F620','\U0001F628','\U0001F632']

# Constants.
CLASS_NUMBER = 6
K            = 10
SPLIT        = 5

"""
Class: Classifier

A Binary Naive Bayes "Emotional Analysis" classifier.

Implementation:
    Author: Ellie Moore

Training Data Used:
    https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset
    Author: Bhavik Jikadara
    License: https://creativecommons.org/licenses/by/4.0/
    Changes Made: None

References:
    https://web.stanford.edu/~jurafsky/slp3/4.pdf
    Authors: Daniel Jurafsky & James H. Martin

"""
class Classifier:

    def __init__(self):
        # If not testing, classify. 
        try:    
            # Load the serialized model.     
            with open("log_prior.pickle", "rb") as f:
                self.log_prior = pickle.load(f)
            with open("setV.pickle", "rb") as f:
                self.setV = pickle.load(f)
            with open("log_likelihood.pickle", "rb") as f:
                self.log_likelihood = pickle.load(f)
        except:    
            # Generate a new model. 
            try:
                with open('emotions.csv', "rb") as f:
                    # Read the file.
                    samples = str(f.read())

                    # Split into samples.
                    samples = samples.split('\\r\\n')
                    samples = samples[1:-1]
            except:
                print("Error: missing dataset file.")
                return

            self.fold(samples)

            with open("log_prior.pickle", "wb") as f:
                pickle.dump(self.log_prior, f)
            with open("setV.pickle", "wb") as f:
                pickle.dump(self.setV, f)
            with open("log_likelihood.pickle", "wb") as f:
                pickle.dump(self.log_likelihood, f)

        # GUI
        t = tk.Tk()
        t.geometry("750x750")
        t.resizable(width=False, height=False)
        f = tf.Font(family="Arial", size=50)
        self.label = tk.Label(t, text=f"Vibe Check", font=f)
        self.label.pack(pady=10)
        self.textbox = tk.Text(t, height=32, width=80)
        self.textbox.pack()
        self.textbox.bind("<KeyRelease>", self.update)
        self.label = tk.Label(t, text=CLASS_EMOJIS[1], font=f)
        self.label.pack(pady=10)
        t.mainloop()
        
    def update(self, event):
        # Update the GUI.
        s = self.textbox.get("1.0", tk.END).strip()
        c = self.classify(s)
        self.label.config(text=f"{CLASS_EMOJIS[c]} | {CLASS_LABELS[c]}")

    def fold(self, samples):
        # Bags for each class.
        bag = [{} for _ in range(CLASS_NUMBER)]

        # Number of samples per class.
        class_count = [0 for _ in range(CLASS_NUMBER)]

        # Log likelihoods for each class.
        self.log_likelihood = [{} for _ in range(CLASS_NUMBER)]

        # Model vocabulary from training data.
        vocabulary = []

        # Number of samples.
        sample_count = len(samples)

        # Iterate through samples.
        i = 0
        for sample in samples:
            # Split sample into features and class.
            u = sample.split(',')
            features = u[0]
            class_num = int(u[1])

            # Increment the sample count for 
            # this class.
            class_count[class_num] += 1

            # split feature string into individual features (words).
            words = features.split(" ")

            # Set of words we've seen for this feature string.
            seen = set()

            # Iterate through the words in the feature string.
            for w in words:

                # Filter stop words.
                if w in stopwords:
                    continue

                # Skip words we've seen (binary naive bayes).
                if w in seen:
                    continue

                # We've now seen this word.
                seen.add(w)

                # If the word is already in the bag for this class,
                # increment the frequency.
                if w in bag[class_num]:
                    bag[class_num][w] += 1

                # If the word isn't already in the bag for this class,
                # Add it to the vocabulary and bag with frequency=1.
                else:
                    vocabulary.append(w)
                    bag[class_num][w] = 1

            i += 1

        # Calculate the priors for each class.
        self.log_prior = np.asarray(class_count, dtype=float)
        self.log_prior /= sample_count
        self.log_prior = np.log(self.log_prior)

        # Calculate the likelihood for each word in each class.
        sig = [0 for _ in range(CLASS_NUMBER)]
        for c in range(CLASS_NUMBER):
            for w in vocabulary:
                cnt = bag[c][w] if w in bag[c] else 0
                sig[c] += cnt + 1

        for c in range(CLASS_NUMBER):
            for w in vocabulary:
                cnt = bag[c][w] if w in bag[c] else 0
                self.log_likelihood[c][w] = np.log((cnt + 1)/sig[c])

        # Convert the vocabulary list to a set.
        self.setV = set(vocabulary)

    def classify(self, s):
        # Remove punctuation.
        str = re.sub(r'[^\w\s]', '', s)
        str = str.replace('\n', '')

        # Convert to lowercase.
        str = str.lower()

        # Split into tokens.
        str = str.split(' ')

        # Classify.
        mx = -math.inf
        class_num = -1
        for c in range(CLASS_NUMBER):
            lp = self.log_prior[c]

            seen = set()
            for w in str:
                w = w.strip()                

                if w in stopwords:
                    continue

                if w in seen:
                    continue
                
                seen.add(w)

                if w in self.setV:
                    lp += self.log_likelihood[c][w]

            if lp > mx:
                mx = lp
                class_num = c

        return class_num

"""
Main
"""
if __name__ == "__main__":
    c = Classifier()
