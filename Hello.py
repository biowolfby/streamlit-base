# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# required libraries
import os
import glob
import streamlit as st
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np

#methods
cs = ["Naive Bayes","SVM"]
classification_space = st.sidebar.selectbox("Pick a classification Method:", cs)

st.write("Results")
st.write('Dataset details here:')
st.write("Twenty Newsgroup dataset chosen. It contains about 18000 different posts from newspapers and 20 different topics")

if st.sidebar.button('Classify'):
    if classification_space == "Naive Bayes":
        trainData = fetch_20newsgroups(subset='train', shuffle=True)
        st.write("Naive Bayes selected")
        classificationPipeline = Pipeline([('bow', CountVectorizer()), ('vector', TfidfTransformer()), ('classifier', MultinomialNB())])
        classificationPipeline = classificationPipeline.fit(trainData.data, trainData.target)
        test_set = fetch_20newsgroups(subset='test', shuffle=True)
        dataPrediction = classificationPipeline.predit(test_set.data)
        st.write("Accuracy of Naive Bayes:")
        st.write(np.mean(dataPrediction == test_set.target))

    if classification_space == "SVM":
        trainData = fetch_20newsgroups(subset='train', shuffle=True)
        st.write("SVM selected")
        classificationPipeline = Pipeline([('bow', CountVectorizer()), ('vector', TfidfTransformer()), ('classifier', SGDClassifier(loss='hing', penalty='11', alpha=0.0005, l1_ratio=0.17))])
        #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        classificationPipeline = classificationPipeline.fit(trainData.data, trainData.target)
        test_set = fetch_20newsgroups(subset='test', shuffle=True)
        dataPrediction - classificationPipeline.preduct(test_set.data)
        st.write("SVM:")
        st.write(np.mean(dataPrediction == test_set.target))
