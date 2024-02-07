word2vec_cbow_path = '../embedding_models/word2vec.model-cbow-size=200-window=5.bin'
glove_path='../embedding_models/gensim_glove_vectors.txt'
stopword_path = '../datasets/stopwords.txt'
dataset_path = '../datasets/task_1402.11.15.csv'
dictionary_path = '../datasets/wiki_fa_80k.txt'
csv_column_name='عنوان'
csv_column_name_to='job'

host = '127.0.0.1'
port = 27017
db_name = 'nlp'
collection_name_csv_to_db='new_version_of_cleaned_dataset'
all_syn_collection_names = 'all_syn'
all_encoded_collection_names = 'syn_encoded'

#%%