from const.constance import *
from maiin.maiin import *


class testtargert:
    def __init__(self):
        self.main_class = Main('final_test', test_word2vec_cbow_path, test_stopword_path)

    def syned_dataset(self):
        self.main_class.finall_proccess(test_dataset_path, test_db, test_csv_column_name, test_csv_column_name_to,
                                        test_syn, test_encoded)

    def test(self, text):
        return self.main_class.result(text)
