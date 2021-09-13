from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

text = """
There are no more barriers to cross. All I have in common with the uncontrollable and the insane, the vicious and the 
evil, all the mayhem I have caused and my utter indifference toward it I have now surpassed. My pain is constant and 
sharp, and I do not hope for a better world for anyone. In fact, I want my pain to be inflicted on others. I want no 
one to escape. But even after admitting this, there is no catharsis; my punishment continues to elude me, and I gain no 
deeper knowledge of myself. No new knowledge can be extracted from my telling. This confession has meant nothing.
"""


def main():
    tokenized_sentences = [word_tokenize(t) for t in sent_tokenize(text)]
    words = [word for sentence in tokenized_sentences for word in sentence]
    words = [word.lower() for word in words if word.isalpha()]
    stop_words = stopwords.words('english')
    filtered_words = [w for w in words if w not in stop_words]
    ps = PorterStemmer()
    ps_stemmed = [ps.stem(word) for word in filtered_words]
    ls = LancasterStemmer()
    ls_stemmed = [ls.stem(word) for word in filtered_words]
    ss = SnowballStemmer('english')
    ss_stemmed = [ss.stem(word) for word in filtered_words]
    print("Porter:\n", ps_stemmed)
    print("Lancaster:\n", ls_stemmed)
    print("Snowball:\n", ss_stemmed)


if __name__ == '__main__':
    main()
