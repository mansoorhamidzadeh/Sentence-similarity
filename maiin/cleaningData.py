import hazm
import pandas as pd
import regex as re
from cleantext import clean


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

    def cleaned_text(self):
        dataset = self.drop_row_with_special_word()
        dataset['cleaned_text'] = dataset[self.csv_column_name_to].apply(self.clean_text)
        dataset.drop(columns={self.csv_column_name_to}, inplace=True)
        return dataset

    def final_dataset(self):
        dataset = self.cleaned_text()
        dataset['number_of_words'] = dataset.cleaned_text.apply(lambda x: len(x.split()))
        dataset.drop(dataset[dataset['number_of_words'] < 3].index, inplace=True)
        dataset.drop(columns={'number_of_words'}, inplace=True)
        dataset.drop_duplicates(inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        return dataset

    def save_to_csv(self, path, name_format):
        dataset = self.final_dataset()
        dataset.to_csv(f'{path}{name_format}', index=False, encoding='utf-8')
