import warnings

warnings.filterwarnings('ignore')

from gensim.models import KeyedVectors
import numpy as np
from farsi_tools import stop_words
from hazm import word_tokenize, Lemmatizer
from pymongo import MongoClient
from sentence_transformers import util
from symspellpy import SymSpell, Verbosity
from tqdm import tqdm
import hazm
import pandas as pd
import regex as re
from cleantext import clean

lemmatizer = Lemmatizer()


# %%

class CleaningData:
    def __init__(self, dataset_file, column_name, csv_column_name_to):
        self.dataset_file = dataset_file
        self.column_name = column_name
        self.csv_column_name_to = csv_column_name_to

    def read_dataset(self):
        dataset = pd.read_csv(self.dataset_file)
        dataset = dataset[[self.column_name]]
        dataset.rename(columns={self.column_name: self.csv_column_name_to}, inplace=True)
        return dataset

    def drop_row_with_special_word(self):
        dataset = self.read_dataset()
        dataset.drop(dataset[dataset[self.csv_column_name_to].str.contains('تولد')].index, inplace=True)
        dataset.dropna(inplace=True)
        dataset.drop_duplicates(inplace=True)
        dataset.reset_index(drop=True, inplace=True)
        return dataset

    def clean_text(self, text):
        text = clean(text,
                     fix_unicode=True,
                     to_ascii=False,
                     no_numbers=True,
                     # no_emoji=True,
                     no_digits=True,
                     no_punct=True,
                     no_emails=True,
                     replace_with_digit='',
                     replace_with_email='',
                     replace_with_phone_number='',
                     replace_with_punct='')
        text = re.sub(r'[\u200c]', '', text)
        text = text.replace('<number>', ' ')
        text = hazm.word_tokenize(text)
        text = ' '.join(text)

        return text

    def clead_dataset(self):
        dataset = self.drop_row_with_special_word()
        dataset['cleaned_text'] = dataset[self.csv_column_name_to].apply(self.clean_text)
        dataset.drop(columns={self.csv_column_name_to}, inplace=True)
        return dataset

    def final_dataset(self):
        dataset = self.clead_dataset()
        dataset['number_of_words'] = dataset.cleaned_text.apply(lambda x: len(x.split()))
        dataset.drop(dataset[dataset['number_of_words'] < 3].index, inplace=True)
        dataset.drop(columns={'number_of_words'}, inplace=True)
        dataset.drop_duplicates(inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        return dataset

    def save_to_csv(self, path, name_format):
        dataset = self.final_dataset()
        dataset.to_csv(f'{path}{name_format}', index=False, encoding='utf-8')


# %%

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
Loadstatic().load_word2vec_model(word2vec_cbow_path)


# %%
class ConnectDatabase:
    def __init__(self, database_name):
        self.database_name = database_name

    def context_mongo(self, collection_name):
        client = MongoClient(host=host, port=port)
        client_my = client[self.database_name]
        my_collection = client_my[collection_name]
        return my_collection

    def read_dataset_csv(self, _dataset_path, _csv_column_name, _csv_column_name_to):
        df = CleaningData(_dataset_path, _csv_column_name, _csv_column_name_to).final_dataset()
        return df

    def set_data_mongo(self, _collection_name_csv_to_db, _dataset_path, _csv_column_name, _csv_column_name_to):
        my_pd_data = self.read_dataset_csv(_dataset_path, _csv_column_name, _csv_column_name_to)
        my_pd_data = my_pd_data.to_dict(orient='records')
        my_coll = self.context_mongo(collection_name_csv_to_db)
        for i in tqdm(my_pd_data):
            if my_coll.find_one({'cleaned_text': i['cleaned_text']}):
                pass
            else:
                my_coll.insert_one(i)

    def synonyms_to_db(self):
        my_coll = self.context_mongo(collection_name_csv_to_db)
        sentences = [i['cleaned_text'] for i in my_coll.find()[:100]]
        return sentences


# %%
cn = ConnectDatabase(database_name=db_name)
cn.set_data_mongo(collection_name_csv_to_db, dataset_path, csv_column_name, csv_column_name_to)


# %%
class Main:

    def __init__(self, database_name, embeding, stopword):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=8)
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.model = Loadstatic().load_glove_model(embeding)
        self.stop_word = Loadstatic().load_stop_words(stopword)
        self.client = ConnectDatabase(database_name)

    def word_embedding_method(self, sentence):
        try:
            encoded_word_list = []
            for i in self.preprocess(sentence):
                if i in self.model:
                    encoded_word_list.append(self.model[i])
                else:
                    continue
            if encoded_word_list is None:
                return None
            else:
                return np.mean(encoded_word_list, axis=0).tolist()
        except KeyError as e:
            return None

    def preprocess(self, raw_text):
        words = word_tokenize(raw_text)  # split a sentence bertokenizer words and return an array
        words = [i for i in words if i not in self.stop_word]
        spell_checker_list = []
        for i in range(len(words)):  # iterate in word array for checking spell
            if not self.sym_spell.lookup(words[i], Verbosity.ALL,
                                         max_edit_distance=3):  # if word not exists ignore the word
                continue
            else:
                word_matched = self.sym_spell.lookup(words[i], Verbosity.ALL, max_edit_distance=3)
                for i in word_matched[:1]:  # take first similar word bertokenizer our incorrect word
                    spell_checker_list.append(i)
        # lemmatize = [lemmatizer.lemmatize(word) for word in words]

        split_lemm_words = []
        for i in words:
            if "#" not in i:
                split_lemm_words.append(i)
            else:
                split_lemm_words.extend(i.split("#"))
        clean_words = list(
            set([w for w in split_lemm_words if w not in self.stop_word]))  # remove some word like "و ,با, ..."

        return clean_words

    def calculate_similarity_of_words(self, sentence1, sentence2):
        sentence_1 = self.preprocess(sentence1)

        sentence_word_similarity = {}
        for i in sentence_1:
            get_similarity = self.model.most_similar_cosmul(i, topn=20)
            lemm = [lemmatizer.lemmatize(word[0]) for word in get_similarity]
            cleaning_words = [re.sub(r'[\u200c]', '', word) for word in lemm]
            sentence_word_similarity[i] = list(set(cleaning_words))
        gf_list = {}
        for sent in sentence2:
            sentence_2 = self.preprocess(sent)
            reformed_sentence = sentence_2.copy()
            for i, j in sentence_word_similarity.items():
                for x in sentence_2:
                    if x in j:
                        reformed_sentence = [i if word == x else word for word in reformed_sentence]
            gf_list = {**gf_list, **{sent: ' '.join(reformed_sentence)}}
        return gf_list

    def find_upload(self, sentences):
        all_predicted_sentences = {}
        for i in tqdm(sentences):
            tokenized = self.preprocess(i)
            if len(tokenized) > 0:
                ref_with_sim = {}
                for word in tokenized:
                    sim_words_array = []
                    try:
                        get_similarity = self.model.most_similar_cosmul(word, topn=10)
                        for similarity in get_similarity:
                            sim_words_array.append(similarity[0])
                    except:
                        pass
                    ref_with_sim = {**ref_with_sim, **{word: sim_words_array}}
                maked_sentence = []
                for tokenized_word in self.preprocess(i):
                    for l, m in ref_with_sim.items():
                        if l == tokenized_word:
                            for ih in m:
                                maked_sentence.append(str(i).replace(l, ih))
                all_predicted_sentences = {**all_predicted_sentences, **{i: maked_sentence}}
            else:
                continue
        return all_predicted_sentences

    def synonyms_to_db(self):
        my_coll = self.client.context_mongo(collection_name_csv_to_db)
        all_syn = self.client.context_mongo(all_syn_collection_names)
        sentences = [i['cleaned_text'] for i in my_coll.find()[:150]]
        for i, j in self.find_upload(sentences).items():
            if all_syn.find_one({'name': i}):
                pass
            else:
                all_syn.insert_one({
                    "name": i,
                    "encoded": j
                })

    def encod_to_db(self):
        my_coll = self.client.context_mongo(all_syn_collection_names)
        syn_encoded = self.client.context_mongo(all_encoded_collection_names)
        for i in tqdm(my_coll.find({}, {"_id": False})):
            if syn_encoded.find_one({'text': i['name']}):
                pass
            else:
                syn_encoded_list = []
                for m in i['encoded']:
                    syn_encoded_list.append(np.mean([self.word_embedding_method(m)], axis=0).tolist())
                if syn_encoded_list:
                    syn_encoded.insert_one({
                        'text': i['name'],
                        'generated_sent': i['encoded'],
                        'mean_encoded': syn_encoded_list

                    })

    def result(self, ref):
        syn_encoded = self.client.context_mongo('syn_encoded_tolied')

        vector_1 = np.mean([self.word_embedding_method(ref)], axis=0)

        res = {}
        for i in tqdm(syn_encoded.find({}, {"_id": False})[:100]):
            result = util.cos_sim(vector_1.tolist(), [c for c in i['mean_encoded'] if type(c) != float])
            res[i['text']] = np.mean(sorted(result[0].detach().numpy(), reverse=True)[:5])
            # res[i['text']] = result[0].detach().numpy()
        return list(sorted(res.items(), key=lambda item: item[1], reverse=True))[:15]
# %%
