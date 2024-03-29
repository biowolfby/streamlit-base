import streamlit as st
# From https://github.com/argosopentech/argos-translate
#TextBlob for sentiment Analysis
from textblob import TextBlob
import argostranslate.package
import argostranslate.translate
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

st.write("Translator app")
st.write("You can translate into these languages: Chinese, Spanish, German, French, and Russian")

st.write("Enter text and choose your language:")

text = st.text_input("Enter input:", "")

#methods
#TASK 1

#after you add options the sidebar would show 4 languages

language_codes = {
    "Chinese": "zh",
    "Spanish": "es",
    "German": "de",
    "Russian": "ru"
}

classification_space = st.radio("Language to be translated into:", list(language_codes.keys()))
option = language_codes.get(classification_space, '')


if st.button('Translate') and option:
    from_code = "en"
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == option, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())
    translated_text = argostranslate.translate.translate(text, from_code, option)
    Pronounce = argostranslate.translate.translate(text, from_code, option)
    
    original_sentiment = TextBlob(text).sentiment
    translated_sentiment = TextBlob(translated_text).sentiment

    st.write("Original Text Sentiment:", original_sentiment)
    st.write("Translated Text Sentiment:", translated_sentiment)
    st.write("Translated Text:", translated_text)
    st.write(Pronounce)

