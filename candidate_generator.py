from rank_bm25 import BM25Okapi
import fasttext
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA
import json
import spacy
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import *
import time
from collections import defaultdict
import concurrent.futures
from itertools import repeat

num_of_candid=1250

def tf_idf(articles,meshWords):
    print('Running tf-tdf')
    mentionSetList=[]
    nlp = spacy.load("en_core_sci_sm")
    tfif_char_vectorizer = TfidfVectorizer(decode_error='ignore', stop_words='english',
                                           analyzer='char', ngram_range=(2, 5),
                                           max_features=100000)
    n=0
    umls_tfidf = tfif_char_vectorizer.fit_transform(meshWords).tocsr()

    dict_words ={}

    for article in articles:
        n+=1
        mentions=[]
        mentionSet = set()
        doc = nlp(article)
        for ent in doc.ents:
            mentions.append(str(ent))
            if str(ent) not in dict_words:
                dict_words[str(ent)] = []

        mention_tfidf = tfif_char_vectorizer.transform(mentions).tocsr()


        shard_start = 0
        delta = mention_tfidf.shape[0]
        keep_scores = []
        topk_hits = defaultdict(int)
        topk_misses = defaultdict(int)
        for shard_num in range(1):
            shard_end = shard_start + delta

            pair_scores = pairwise_distances(mention_tfidf[shard_start:shard_end,:], umls_tfidf, metric='cosine', n_jobs=-1)
            arg_mins = np.argpartition(pair_scores, num_of_candid, axis=1)

            for row in range(pair_scores.shape[0]):
                hit = False
                cur_mins = arg_mins[row]
                listing=[]
                for topK in range(num_of_candid):
                    match_idx = arg_mins[row, topK]
                    listing.append((meshWords[match_idx],pair_scores[row,match_idx]))
                    mentionSet.add(meshWords[match_idx])
                # if dict_words[mentions[row]] == []:
                #
                #     sorted(listing, key=lambda x: x[1])
                #
                #     new_list = [x[0] for x in listing]
                #     dict_words[mentions[row]] = new_list

        mentionSetList.append(mentionSet)
    # print(dict_words)
    # return dict_words
    return mentionSetList

def bm_candidate_gen(articles,meshWords):
    mentionSetList=[]
    for article in articles:

        corpus = [str(m) for m in meshWords]
        bm25 = BM25Okapi(corpus)
        mentions=[]
        mentionSet = set()
        doc = nlp(article)
        for ent in doc.ents:
            mentions.append(str(ent))

        corpus = [str(m) for m in meshWords]
        for word in mentions:


            mentionSet = mentionSet.union(set(bm25.get_top_n(word, corpus, n=num_of_candid)))
        mentionSetList.append(mentionSet)
    return mentionSetList


def fasttext_candidate_gen(articles,meshWords):
    print('Running fasttext')
    model_fasttext = fasttext.load_model('model_fasttext.bin')
    mentionSetList=[]
    mesh_vectors_fasttext=[]
    for word in meshWords:
        mesh_vectors_fasttext.append(model_fasttext.get_word_vector(word))

    tree = NearestNeighbors(n_neighbors=num_of_candid,algorithm='ball_tree').fit(mesh_vectors_fasttext)
    mesh_np = np.array(meshWords)
    n=0
    for article in articles:
        if n%100==0:
            print(n)
        mentions=[]
        mentionSet = set()
        doc = nlp(article)
        for ent in doc.ents:
            mentions.append(str(ent).lower())
        t1=time.time()
        word_vect = [model_fasttext.get_word_vector(word_candid) for word_candid in mentions]
        t2=time.time()
        ind = tree.kneighbors(word_vect,return_distance=False)

        words = [mesh_np[ind[a]] for a in range(len(ind))]
        # words = mesh_np[]
        t3=time.time()
        # print(t3-t2)
        word_set = [set(w) for w in words]

        mentionSet = mentionSet.union(*word_set)
        mentionSetList.append(mentionSet)
        n+=1
    return mentionSetList


# Splitting articles
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out
nlp = spacy.load("en_core_sci_sm")
if __name__ == '__main__':

    # Parameters
    num_of_candid=1250
    num_cores_used = 8


    file = open("mesh_2018_ID.txt","r")
    mesh = file.readlines()
    meshWords = []
    meshID = []
    for n, line  in enumerate(mesh):
        lineArr = line.split("=")
        word = lineArr[0].lower()
        meshWords.append(word)

    # Creates a dictionary to map words to index
    wordToIdx = {}
    for n, word in enumerate(meshWords):
        wordToIdx[word] = n

    #Loading in all of the training and testing data
    testing_articles=[]
    test_mesh=[]
    articles=[]
    groundtruth_mesh=[]
    with open('data_medium.json') as json_file:
        data= json.load(json_file)
        for d in data['train']:
            articles.append(d['article'])
            groundtruth_mesh.append(d['mesh_labels'])
        for d in data['test']:
            testing_articles.append(d['article'])
            test_mesh.append(d['mesh_labels'])

    print("Finished loading in data")





    split_articles = chunkIt(articles,num_cores_used)
    split_test_articles = chunkIt(testing_articles,num_cores_used)


    mesh_l_train = []
    for n in range(len(split_articles)):
        mesh_l_train.append(meshWords)
    meshWords=mesh_l_train

    # Generating candidates through tf idf
    candidate_mesh_list_train = []
    candidate_mesh_list_test = []
    print("Running tf-idf candidate generator")


    t1=time.time()
    dict_train = {}
    dict_test = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(tf_idf,split_articles,meshWords)
        for result in results:
            candidate_mesh_list_train+=result
            # dict_train = {**result, **dict_train}
        print('Generating Testing articles')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(tf_idf,split_test_articles,meshWords)
        for result in results:
            candidate_mesh_list_test+=result
            # dict_test = {**result, **dict_test}

    set_avg= [len(s) for s in candidate_mesh_list_train]
    print("tf-idf avg mesh label size:",sum(set_avg)/len(set_avg))



    t2=time.time()
    print("Finished tf-idf candidate generator| Time taken:",t2-t1)

    candidate_mesh_list_train = [list(s) for s in candidate_mesh_list_train]
    candidate_mesh_list_test = [list(s) for s in candidate_mesh_list_test]

    data={}
    data['train'] = candidate_mesh_list_train
    data['test'] = candidate_mesh_list_test
    with open('candidate_data.json','w') as outfile:
        json.dump(data,outfile)
    print("Created file candidate_data.json")



    #Generating candidates through bm25

    candidate_mesh_list_train = []
    candidate_mesh_list_test = []

    print("Running bm25 candidate generator")
    t1=time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(bm_candidate_gen,split_articles,meshWords)
        for result in results:
            candidate_mesh_list_train+=result
        print('Generating Testing articles')
        results = executor.map(bm_candidate_gen,split_test_articles,meshWords)
        for result in results:
            candidate_mesh_list_test+=result

    set_avg= [len(s) for s in candidate_mesh_list_train]
    print("bm25 avg mesh label size:",sum(set_avg)/len(set_avg))

    t2=time.time()
    print("Finished bm25 candidate generator| Time taken:",t2-t1)

    candidate_mesh_list_train = [list(s) for s in candidate_mesh_list_train]
    candidate_mesh_list_test = [list(s) for s in candidate_mesh_list_test]
    data={}
    data['train'] = candidate_mesh_list_train
    data['test'] = candidate_mesh_list_test
    with open('candidate_data_bm25.json','w') as outfile:
        json.dump(data,outfile)
    print("Created file candidate_data_bm25.json")


    #Generating through fasttext

    mesh_vectors_fasttext =[]
    print("Finished training fasttext")

    candidate_mesh_list_train = []
    candidate_mesh_list_test = []
    print("Running fasttext candidate generator")
    t1=time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(fasttext_candidate_gen,split_articles,meshWords)

        for result in results:
            candidate_mesh_list_train+=result
        print('Generating Testing articles')
        results = executor.map(fasttext_candidate_gen,split_test_articles,meshWords)
        for result in results:
            candidate_mesh_list_test+=result
    set_avg= [len(s) for s in candidate_mesh_list_train]
    print("fasttext avg mesh label size:",sum(set_avg)/len(set_avg))
    t2=time.time()
    print("Finished fasttext candidate generator| Time taken:",t2-t1)
    candidate_mesh_list_train_update = [list(s) for s in candidate_mesh_list_train]
    candidate_mesh_list_test_update = [list(s) for s in candidate_mesh_list_test]

    data={}
    data['train'] = candidate_mesh_list_train_update
    data['test'] = candidate_mesh_list_test_update
    with open('candidate_data_fasttext.json','w') as outfile:
        json.dump(data,outfile)

    print("Created file candidate_data_fasttext.json")
    print("Finished")
