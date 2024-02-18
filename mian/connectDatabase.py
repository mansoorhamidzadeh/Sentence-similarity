from pymongo import MongoClient
from const.constance import *
from mian.cleaningData import *
from tqdm import tqdm

class ConnectDatabase:
    def __init__(self, database_name):
        self.database_name = database_name


    def context_mongo(self,collection_name):
        client = MongoClient(host=test_host, port=test_port)
        client_my = client[self.database_name]
        my_collection =client_my[collection_name]
        return my_collection

    def read_dataset_csv(self, _dataset_path, _csv_column_name, _csv_column_name_to):
        df = CleaningData(_dataset_path, _csv_column_name, _csv_column_name_to).final_dataset()
        return df

    def set_data_mongo(self, _collection_name_csv_to_db, _dataset_path, _csv_column_name, _csv_column_name_to):
        my_pd_data = self.read_dataset_csv(_dataset_path, _csv_column_name, _csv_column_name_to)
        my_pd_data = my_pd_data.to_dict(orient='records')
        my_coll = self.context_mongo(_collection_name_csv_to_db)
        for i in tqdm(my_pd_data):
            if my_coll.find_one({'cleaned_text': i['cleaned_text']}):
                pass
            else:
                my_coll.insert_one(i)

