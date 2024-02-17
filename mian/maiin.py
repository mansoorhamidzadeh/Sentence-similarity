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
from const.constance import *
import hazm
import pandas as pd
import regex as re
from cleantext import clean
from mian.cleaningData import *
from mian.connectDatabase import *
from mian.loadStatic import *
lemmatizer = Lemmatizer()
class Main:


    def __init__(self,database_name,embeding,stopword):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=8)
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.model = Loadstatic().load_glove_model(embeding)
        self.stop_word = Loadstatic().load_stop_words(stopword)
        self.client=ConnectDatabase(database_name)
    def clean_data_to_db(self):
        self.client.set_data_mongo(collection_name_csv_to_db,
                                   '../datasets/xx.csv',
                                   csv_column_name,csv_column_name_to)
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
        sentences = [i['cleaned_text'] for i in my_coll.find()[:13]]
        sen_list=[]
        for i in sentences:
            if all_syn.find_one({'name': i}):
                pass
            else:sen_list.append(i)

        for i, j in self.find_upload(sen_list).items():
                all_syn.insert_one({
                    "name": i,
                    "encoded": j
                })

    def encod_to_db(self):
        my_coll = self.client.context_mongo(all_syn_collection_names)
        syn_encoded =  self.client.context_mongo(all_encoded_collection_names)
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
    def finall_proccess(self):
        self.clean_data_to_db()
        self.synonyms_to_db()
        self.encod_to_db()
    def result(self,ref):
        syn_encoded = self.client.context_mongo('syn_encoded_tolied')

        vector_1 = np.mean([self.word_embedding_method(ref)], axis=0)

        res = {}
        for i in tqdm(syn_encoded.find({}, {"_id": False})[:100]):

            result = util.cos_sim(vector_1.tolist(), [c for c in i['mean_encoded'] if type(c) != float])
            res[i['text']] = np.mean(sorted(result[0].detach().numpy(), reverse=True)[:5])
            # res[i['text']] = result[0].detach().numpy()
        return list(sorted(res.items(), key=lambda item: item[1], reverse=True))[:15]
#%%
