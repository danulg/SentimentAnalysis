import dill
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from dataloader import IMDBDataSet
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

if __name__ == '__main__':
    imdb = IMDBDataSet()
    text, labels = imdb.reviews(ret_val=True)

    train_dt, test_dt, train_lb, test_lb = train_test_split(text, labels, test_size=0.2, random_state=42)
    bag_of_words_est = Pipeline([('cvt', CountVectorizer(min_df=0.005)), ('mb', MultinomialNB())])

    bag_of_words_est.fit(train_dt, train_lb)
    dill.dump(bag_of_words_est, open('bag_of_words_est', 'wb'))

    y_pred = bag_of_words_est.predict(test_dt)

    error = accuracy_score(y_pred, test_lb)

    print(error)

