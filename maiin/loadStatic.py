import regex as re
from farsi_tools import stop_words
from gensim.models import KeyedVectors


class Loadstatic:
    def __init__(self):
        pass

    def load_stop_words(self, _stopword_path):
        with open(_stopword_path, encoding="utf8") as f:
            stop = f.readlines()
        stop_word = [word.replace('\n', '') for word in stop]
        stop_word = [re.sub('[\\u200c]', ' ', word) for word in stop_word]
        stop_word.extend(stop_words())
        return stop_word

    def load_glove_model(sekf, _glove_file):
        print("loading glove model")
        model = KeyedVectors.load_word2vec_format(_glove_file, binary=False)
        print(f"loaded glove model , {len(model)}")
        return model

    def load_word2vec_model(self, _word2vec_cbow_path):
        print('loading word2vec model')
        word2vec_model = KeyedVectors.load_word2vec_format(_word2vec_cbow_path, binary=True)
        print(f"loaded word2vec model,{len(word2vec_model)}")
        return word2vec_model

# %%
