import hazm
import regex as re
from cleantext import clean


class cleanSentence:
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
