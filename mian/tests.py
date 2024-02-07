# %%
class Testing:
    def __init__(self):
        self.model = Loadstatic().load_word2vec_moel(word2vec_cbow_path)

    def find_synonyms_of_word(self, ref, num):
        result = self.model.most_similar_cosmul(ref, topn=num)
        return result

    def calculate_similarity_of_words(sentence1, sentence2):
        sentence_1 = preprocess(sentence1)
        sentence_word_similarity = {}
        for i in sentence_1:
            get_similarity = word2vec_model.most_similar_cosmul(i, topn=20)
            lemm = [lemmatizer.lemmatize(word[0]) for word in get_similarity]
            cleaning_words = [re.sub(r'[\u200c]', '', word) for word in lemm]
            sentence_word_similarity[i] = list(set(cleaning_words))

        sentence_2 = preprocess(sentence2)

        reformed_sentence = sentence_2.copy()
        for i, j in sentence_word_similarity.items():
            for x in sentence_2:
                if x in j:
                    reformed_sentence = [i if word == x else word for word in reformed_sentence]
        gf_list = ' '.join(reformed_sentence)
        return gf_list

    def single_sim(sentence1, sentence2):
        vector_1 = np.mean([word_embedding_method(sentence1)], axis=0)
        result_pass = calculate_similarity_of_words(sentence1, sentence2)

        op = {}
        if len(result_pass.split(' ')) > 1:
            op[result_pass] = np.mean([word_embedding_method(result_pass)], axis=0)
        else:
            if type(word_embedding_method(result_pass)) == float:
                pass
            else:
                op[result_pass] = word_embedding_method(result_pass)
        result_dict = {}
        result = util.cos_sim(vector_1, np.array([j if i else None for i, j in op.items()], dtype=float))[0]
        return result
