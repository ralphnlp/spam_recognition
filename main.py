import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
#nltk.download('punkt')

def load_data(data_path):
    
    data = pd.read_csv(data_path, encoding='ISO-8859-1')
    labels, texts = data['v1'].tolist(), data['v2'].tolist()

    bincount, classes = {}, set(labels)
    for cls in classes:
        bincount[cls] = 0
    for label in labels:
        bincount[label] += 1

    return labels, texts, (bincount)


def cleaning_texts(texts):
    
    def cleaning_text(text):
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha()]
        cleaned_text = ' '.join(tokens)
        return cleaned_text
    
    cleaned_texts = []
    for text in tqdm(texts):
        cleaned_text = cleaning_text(text)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts


if __name__ == '__main__':
    
    data_path = './spam.csv'
    labels, texts, bincount = load_data(data_path)
    print(f"[INFO] Label bin_count = {bincount}, Total samples = {sum(bincount.values())}")
    x_train, x_test, label_train, label_test = train_test_split(texts, labels, test_size=0.2)
    print(f'[INFO] train shape = {len(x_train)}, test shape = {len(x_test)}')

    cleaned_x_train = cleaning_texts(x_train)
    countVectorizer = CountVectorizer()
    transformed_x_train = countVectorizer.fit_transform(cleaned_x_train).toarray()
    print(f"[INFO] Matrix count frequency texts = {transformed_x_train.shape}")

    classfier = MultinomialNB()
    print('[INFO] Traing...')
    classfier.fit(transformed_x_train, label_train)

    print('[INFO] Testing...')
    transformed_x_test = countVectorizer.transform(cleaning_texts(x_test))
    predicted_label_test = classfier.predict(transformed_x_test)
    
    print('[INFO] Display result...')
    print(classification_report(label_test, predicted_label_test))



