from flask import Flask
from flask_restful import Api, request
import pandas as pd
import re
from numpy import argmax
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models

app = Flask(__name__)
api = Api(app)

stop_words = set(
    "him than me being don elsewhere from isn't within never fifty this even many can't whos afterwards most because she'd that - he'd couldn't everything that's she's had against four were where nor ours although always amount ] you'd -lsb- until unless ! anything been whatever he's nevertheless couldnt + below have ' every won't besides full several their two where's these fifteen i've thereafter perhaps per around say ?? beforehand latter us t almost done wont otherwise hes myself they're just except or whether hadnt shouldnt you by due call first else his too whereupon hows -lrb- and thats beside !? something formerly wheres either youll [ again mine so thence ; , heres he'll various latterly does you've eleven least might over own they've must they whoever ever whats once anywhere wherever didn't we seem did here's doing no noone some more im see behind having theyve dont havent down which quite one last all toward why others our make forty is empty shes as an along rather if though whose s often back was get he any none 'm becoming up at it's now anyway shouldn't about a himself am < theyre she across part & doesn't haven't @ already should for i'm name both arent how's youre after whereafter whenever front regarding ? towards !! she'll when i'd when's i cannot . everyone twenty of ca re whither we're seems thus her show then ^ same can $ the > has # on do give you're who's 'll wasn't thru themselves my .. who above hereafter next theyll before in hereby whom why's with be mustn't we'll wherein former someone mostly take therefore werent shan't hasnt doesnt % what's during anyone yourself we've out please through wouldnt your let's mustnt among hasn't another became don't isnt move between less -rrb- much off onto hadn't ourselves they'd wouldn't few there's using beyond via those under three to they'll nothing 's thereupon ten throughout ?! upon nine itself such seeming whens ) somehow ### meanwhile hereupon enough into it therein return ought keep whence ` nobody further serious five whereby whys namely cant would theirs other how indeed become somewhere weren't since everywhere * but there only yours whole them without hers lets herself together will -rsb- anyhow well third here shant eight side ... `` very moreover youd i'll whereas top aren't we'd however still what its '' may bottom you'll could yourselves youve thereby theres not also sixty six } wasnt amongst : yet alone made didnt really nowhere \" hence becomes ( seemed { each neither twelve sometimes are hundred put used while sometime herein go".split())


@app.route('/cluster', methods=['GET', 'POST'])
def cluster_api():
    if request.method == 'POST':
        df = pd.DataFrame(pd.read_json(request.data))
        df = online_cluster(df)
        return df.to_json(orient='records')
    return 'wrong request method!'


def online_cluster(df: pd.DataFrame):
    def normalize(word):
        #     Remove non-ASCII characters from list of tokenized words
        #     text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return re.sub(r'[^\w\s]', '', word.lower())

    def tokenize(text):
        return word_tokenize(text)

    def lemma(word):
        lem = WordNetLemmatizer()
        res = lem.lemmatize(word, pos='v')
        if res == word:
            res = lem.lemmatize(word, pos='n')
        return res

    def get_valid_tokens(text):
        tokens = tokenize(text)
        valid_tokens = []
        for token in tokens:
            norm_token = normalize(token)
            if len(norm_token) >= length_lower_bound and norm_token not in stop_words:
                valid_tokens.append(lemma(norm_token))
        return valid_tokens

    texts = []
    length_lower_bound = 3

    for index, row in df.iterrows():
        data_column = 'text'
        texts.append(get_valid_tokens(row[data_column]))

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=len(texts) / 1000, no_above=0.5, keep_n=len(texts) * 50)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda = models.ldamodel.LdaModel(corpus, num_topics=20, id2word=dictionary, passes=40)

    cluster_ids = []
    cluster_label_vectors = []
    for bow in corpus:
        id = argmax(lda[bow], axis=0)[1]
        labels = [row[0] for row in lda.show_topic(id)]
        cluster_ids.append(id)
        cluster_label_vectors.append(labels)
    assert len(texts) == len(df) and len(cluster_label_vectors) == len(df) and len(cluster_ids) == len(df)
    df['cluster_id'] = cluster_ids
    df['cluster_labels'] = cluster_label_vectors
    return df


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
