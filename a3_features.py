import os
import sys
import argparse
import numpy as np
import pandas as pd
# Whatever other imports you need

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from glob import glob
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()


    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.

    def word_counts_n(txt):
        to_df = []
        for count in Counter(txt).most_common():
            if count[0].isalpha():
                to_df.append(count)
            return to_df

    def build_table(samples):
        allfiles = glob("{}/*/*.*".format(args.inputdir), recursive=True)
        column = {}
        filenames = []
        classnames = []
        wordcounts = {}

        for f in allfiles:
            classname = f.split("/")[1]
            filename = f.split("/")[1] + "_" + f.split("/")[2]
            filenames.append(filename)
            classnames.append(classname)

            with open(f, "r") as doc:
                word_n = word_counts_n(doc.read().split(" "))

                if len(word_n) == 0:
                    continue
                for word in word_n:
                    if filename not in wordcounts:
                        wordcounts[filename] = {}
                    wordcounts[filename][word[0]] = word[1]

                    if word[0] not in column:

                        column[word[0]] = []

        for f in allfiles:
            classname = f.split("/")[1]
            filename = f.split("/")[1] + "_" + f.split("/")[2]


            with open(f,"r") as doc:
                word_n = word_counts_n(doc.read().split(" "))
       
                for c in column:
                    if filename in wordcounts and c in wordcounts[filename]:

                        column[c].append(wordcounts[filename][c])
                    else:
                        column[c].append(0)
        
        l = []
        for f in allfiles:

            with open(f,"r") as doc:
                l.append(len(doc.read().split(" ")))

        for name in column:
            if name is not "class" and name is not "filename":

        
                t = column[name]
                res_c = [float(ti)/li for ti,li in zip(t,l)]
                column[name] = res_c
               
        
        df = pd.DataFrame(column)
        svd = TruncatedSVD(n_components=args.dims)
        to_file = svd.fit_transform(df.to_numpy())
        
        X = shuffle(to_file, classnames)
        X_train = to_file[:int(len(to_file)*0.8)]
        X_test = to_file[int(len(to_file)*0.8):]
        y_train = classnames[:int(len(classnames)*0.8)]
        y_test = classnames[int(len(classnames)*0.8):]

        train = pd.DataFrame(X_train)
        train["filename"] = y_train
        train["target"] = ["train"] * len(X_train)

        test = pd.DataFrame(X_test)
        test["filename"] = y_test
        test["target"] = ["test"] * len(X_test)        
        
        return train.append(test)


    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    combined = build_table("enron_sample")
    
    print("Writing to {}...".format(args.outputfile))
    combined.to_csv(path_or_buf=args.outputfile, mode="w", index=False)

    print("Done!")
    
