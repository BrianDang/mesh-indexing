from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import numpy as np
np.random.seed(0)
import sys
import spacy
import json
import nltk
#nltk.download('punkt')

nlp = spacy.load("en_core_sci_sm")


learning_rate = 0.005
mesh_dimension=100
cnn_dimension=100
max_epoch = 100
negative_sampling=500
num_prediction=15
dropout_rate = 0.5
early_stop_count = 3

def breakparagraph(paragraph):
    return sent_tokenize(paragraph)

# Loads all of the mesh words
#'''
file = open("mesh_2018_ID.txt","r")
mesh = file.readlines()
meshWords = []
for n, line  in enumerate(mesh):
    lineArr = line.split("=")
    word = lineArr[0].lower()
    meshWords.append(word)
#'''

words = []
idx = 0
word2idx = {}
vectors=[]

# Must download a file into directory for Glove 100d fro this url https://www.kaggle.com/terenceliu4444/glove6b100dtxt
# Must rename it to the file name 6B100d.txt
with open("6B100d.txt", 'r',encoding= 'utf8') as f:
    for l in f:
        line = l.split()
        word = line[0]

        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
#
glove = {w: vectors[word2idx[w]] for w in words}
print('Done loading in Glove')
sys.stdout.flush()



#Loading in all of the training and testing data
testing_articles=[]
test_mesh=[]
articles=[]
groundtruth_mesh=[]
with open('data.json') as json_file:
    data= json.load(json_file)
    for d in data['train']:
        articles.append(d['article'])
        groundtruth_mesh.append(d['mesh_labels'])
    for d in data['test']:
        testing_articles.append(d['article'])
        test_mesh.append(d['mesh_labels'])
# Loading in all of the candidates for each article in training

with open('candidate_data.json') as json_file:
    data = json.load(json_file)
    candidate_train = [set(d) for d in data['train']]
    #print(len(candidate_train[0]), len(set(candidate_train[0])))
    sys.stdout.flush()
    candidate_test = [set(d) for d in data['test']]
    


matrix_len = len(meshWords)
print('mesh label size:', matrix_len)
weights_matrix = np.zeros((matrix_len+1, mesh_dimension))
words_found = 0
meshWord2Idx = {}
# Creates the word embedding matrix for the MEsH labels 
for i, word in enumerate(meshWords):
    meshWord2Idx[word] = i
    #try: 
    #    weights_matrix[i] = glove[word]
    #    words_found += 1
    #except KeyError:
    #    # If MEsH word is not found, generate a new word embedding 
    weights_matrix[i] = np.random.normal(scale=0.6, size=(mesh_dimension, ))
#For unknown Mesh Words
meshWord2Idx['unk']=len(meshWords)
print(len(meshWord2Idx))




def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

def buildVocab(articles):
    vocabulary=set()
    for article in articles:
        vocabulary.update(set(word_tokenize(article)))
    words_found=0
    vocab2Index = {}
    weights = np.zeros((len(vocabulary)+2, 100))
    for i, word in enumerate(vocabulary):
        vocab2Index[word] = i
        try: 
            weights[i] = glove[word]
            words_found += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.6, size=(100, ))
    vocab2Index['unk']=len(vocabulary)
    weights[len(vocabulary)] = np.random.normal(scale=0.6, size=(100, ))
    vocab2Index['<pad>'] = len(vocabulary)+1
    weights[len(vocabulary)+1] = np.random.normal(scale=0.6, size=(100, ))
    #print('Glove Embeddings found for Vocab: ' + str(words_found) +' out of ' + str(len(vocabulary)))
    return weights, vocab2Index


vocab_weights,vocab2Index = buildVocab(articles)


# Function that outputs targets for hingeloss
# Input: list of candidate entries , list of groundtruth labels
# Output: List of 1 and -1, 1 if is groundtruth label and -1 if not
def labelFinder(candEnt, truthEnt):
    labels = []
    for ent in candEnt:
        if candEnt in truthEnt:
            labels.append[1]
        else:
            labels.append[-1]
    return labels

def meshGenerator(mesh_list):
    list_index = []
    for mesh in mesh_list:
        mesh = mesh.lower()
        if mesh in meshWord2Idx:
            list_index.append(meshWord2Idx[mesh])
        else:
            list_index.append(meshWord2Idx['unk'])
    return torch.LongTensor(list_index)


def helper(groundtruth_mesh, candid_mesh, train=False):
    pos_positions = []
    neg_positions = []
    for n,mesh in enumerate(candid_mesh):
        
        if mesh in groundtruth_mesh:
            pos_positions.append(n)
        else:
            neg_positions.append(n)

    if pos_positions==[]:
        return [],[],[]

    pos_labels_ = np.random.choice(pos_positions, size=negative_sampling)

    neg_labels_ = np.random.choice(neg_positions, size=negative_sampling)
    pos_list = torch.LongTensor(pos_labels_)
    neg_list = torch.LongTensor(neg_labels_)
    return pos_list, neg_list, pos_positions



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

class MeshNN(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers):
        super(MeshNN,self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, inp):
        return self.embedding(inp)

# Constructor takes in the weight matrix for the vocabulary
# Input: One sentence
# Output: Sentence embedding
class CNN(nn.Module):
    def __init__(self,weights_matrix):
        super(CNN,self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.conv0= nn.Conv2d(in_channels = 1,
                             out_channels = int(cnn_dimension / 2),
                             kernel_size = (3,100))
        self.conv1= nn.Conv2d(in_channels = 1,
                             out_channels = int(cnn_dimension / 2),
                             kernel_size = (4,100))
        self.linear = nn.Linear(int(cnn_dimension/2) * 2, mesh_dimension)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, text):
        embedded = self.embedding(text)

        embedded = embedded.unsqueeze(1)

        c0 = F.relu(self.conv0(embedded).squeeze(3))
        c1 = F.relu(self.conv1(embedded).squeeze(3))
        #print(c0.size(), c1.size())
        pool0 = F.max_pool1d(c0, c0.shape[2]).squeeze(2)
        pool1 = F.max_pool1d(c1, c1.shape[2]).squeeze(2) 
        #print(pool0.size(), pool1.size())
        #sys.stdout.flush()
        cat = self.dropout(torch.cat((pool0, pool1), dim = 1))
        return self.linear(cat)

# EnsembleNN Model that takes in two models and a weight matrix: one model to map GLOVe words and another model to encode sentences
# The weight matrix holds the word embeddings for all of the mesh words

class EnsembleNN(nn.Module):
    def __init__(self,MeshModel, sent_embedding_model,weightMatrix,vocabMatrix):
        super(EnsembleNN,self).__init__()
        self.meshModel = MeshModel(weightMatrix,1,1)
        self.sentModel = sent_embedding_model(vocabMatrix)
        # self.lin1 = nn.Linear(768,dimension)

        # forward function takes in a list of sentences and a Long Tensor of indices of mesh labels
    def forward(self, sentences, meshLabels):
        colMatrix = self.meshModel(meshLabels) # num_candidate, mesh_dim
        rowMatrix = self.sentModel(sentences) # num_entity, mesh_dim
        # rowMatrix = self.lin1(torch.FloatTensor(rowMatrix))
        fullMatrix = rowMatrix.mm(colMatrix.t()) # num_entity, num_candidate
        #meanMatrix = torch.logsumexp(fullMatrix,dim=0) 
        meanMatrix = torch.max(fullMatrix,dim=0)[0]
        return meanMatrix
    
    


def calculate_tp_fp(pred_mesh, groundtruth_mesh):
    pred_mesh_set = set(pred_mesh)
    groundtruth_set = groundtruth_mesh
    true_positives = pred_mesh_set.intersection(groundtruth_set)
    length_false_positives = len(pred_mesh_set) - len(true_positives)
    return len(true_positives),length_false_positives
        

def building_right_wrong_array(pred_mesh, groundtruth_mesh, arr):
    for mesh in pred_mesh:
        if mesh in groundtruth_mesh:
            arr.append(1)
        else:
            arr.append(0)

def convert_text_to_id(articles):
    
    articles_wid = []
    for n,article in enumerate(articles):
        sentences = breakparagraph(article)
        # Add <start> and <end> tags around the entities in the sentences
        tokenized_sentences = []
        for sentence in sentences:
            doc = nlp(sentence)
            entities_test = list(doc.ents)
            for ent in entities_test:
                idx = sentence.index(str(ent))
                new_string = sentence[:idx] + ' <start> ' + sentence[idx: idx+len(str(ent))] + ' <end> '+ sentence[idx+len(str(ent)):]
                tokenized_sentences.append(new_string)

        # Tokenize the words in the list of sentences
        word_sentences = []
        max_length = 0
        for sentence in tokenized_sentences:
            toAdd = word_tokenize(sentence)
            word_sentences.append(word_tokenize(sentence))
            if max_length < len(toAdd):
                max_length = len(toAdd)

        # Pads sentences for those that are not of max length
        for sentence in word_sentences:
            if len(sentence)<max_length:
                while len(sentence)!=max_length:
                    sentence.append('<pad>')

        batch_sentences = []
        index_text = []
        for s in word_sentences:
            index_text = []
            for t in s:
                if t in vocab2Index:
                    index_text.append(vocab2Index[t])
                else:
                    index_text.append(vocab2Index['unk'])
            batch_sentences.append(index_text)
        sentences_id = torch.LongTensor(batch_sentences)
        articles_wid.append(sentences_id)
    return articles_wid

training_articles_wid = convert_text_to_id(articles)
testing_articles_wid = convert_text_to_id(testing_articles)


def evaluate(device):
    losses = []
    test_articles = testing_articles_wid
    test_meshlabels = test_mesh
    tp=0
    fp=0
    total_mesh=0
    model.eval()
    with torch.no_grad():
        for n,article in enumerate(test_articles): 
            setHold = meshGenerator(candidate_test[n])

            mesh_list = meshGenerator(test_meshlabels[n])
            out = model(article.to(device), setHold.to(device))
            #out = out[0]
            values, indices = out.topk(num_prediction)
            pred_mesh = [meshWords[n] for n in setHold[indices]]
            mesh_listing = set([word.lower() for word in test_meshlabels[n]])
            pos_list,neg_list,pos_positions = helper(mesh_list,setHold)
            #if n ==5:
            #    print('Values of true mesh labels',out[pos_positions])
            t,f = calculate_tp_fp(pred_mesh,mesh_listing)
            tp+=t
            fp+=f
            total_mesh+=len(mesh_listing)
            
            if len(pos_list)==0:
                continue
            scores = out[pos_list] - out[neg_list]
            loss = criterion(scores, torch.tensor([-1]*negative_sampling).to(device))
            losses.append(loss)

    precision = tp/(tp+fp)
    recall = tp/total_mesh
    print('Average Testing Loss',sum(losses)/len(losses))
    if precision+recall==0:
        return 0,0,0
    else:
        
        micro_score = 2*precision*recall/(precision+recall)
        return micro_score,recall,precision 
    
    


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
sys.stdout.flush()

vocab_weights = torch.FloatTensor(vocab_weights)
torch_weights = torch.FloatTensor(weights_matrix)
# labelIndexes = torch.LongTensor(labelIndexes)
cnn_model = CNN(vocab_weights)
model = EnsembleNN(MeshNN, CNN, torch_weights, vocab_weights).to(device)
# Creating optimizer and loss functions
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.HingeEmbeddingLoss()




    
#candidate_mesh_indices = meshGenerator(list(meshWord2Idx.keys()))
# Training cell
n=0

prev_macro = None
prev_micro=None
early_stop=False
for e in range(max_epoch):
    tp=0
    fp=0
    total_mesh = 0
    losses=[]
    g_tp=0
    g_fp=0
    for n,article in enumerate(training_articles_wid): 

        if n%10==0:
            print(n)
            sys.stdout.flush()
        
        candidate_mesh_indices = meshGenerator(candidate_train[n])
        groundtruth_mesh_indices = meshGenerator(groundtruth_mesh[n])

        pos_list,neg_list,pos_positions = helper(groundtruth_mesh_indices,candidate_mesh_indices,train=True)
        if len(pos_list)==0:
            continue
        model.train()
        out = model(article.to(device), candidate_mesh_indices.to(device))
        #out = out[0]

        scores = out[pos_list] - out[neg_list]
        loss = criterion(scores, torch.tensor([-1]*negative_sampling).to(device))
        losses.append(loss)
        #'''
        values, indices = out.topk(num_prediction)
        pred_mesh = [meshWords[n] for n in candidate_mesh_indices[indices]]
        mesh_listing = set([word.lower() for word in groundtruth_mesh[n]])        
        t,f = calculate_tp_fp(pred_mesh,mesh_listing)
        tp+=t
        fp+=f
        total_mesh+=len(mesh_listing)
        
        candidate_mesh_set = set(candidate_mesh_indices)

        
        candidate_mesh_name_set = set(candidate_train[n])
        groundtruth_mesh_set = set(mesh_listing)
        intersected_set = candidate_mesh_name_set.intersection(groundtruth_mesh_set)

        
        #g_tp += min(len(intersected_set),len(mesh_listing))
        g_tp += min(len(intersected_set),num_prediction)

        #g_fp += max(0,num_prediction-min(len(intersected_set),len(mesh_listing)))
        g_fp += max(0,num_prediction-min(len(intersected_set),num_prediction))
        #'''
        model.zero_grad()
        loss.backward()
        optimizer.step()
        s5=time.time()

        if n==99 and n!=0:

            micro_score, recall, precision = evaluate(device)
            print('Testing Micro score:',micro_score, '| Recall',recall,'| Precision',precision)
            if prev_micro!=None and (micro_score<prev_micro):
                print('Micro score:', micro_score,'previous micro score',prev_micro, '| Recall',recall,'| Precision',precision)
                count+=1
            else:
                count =0
            if count==early_stop_count:
                
                early_stop=True
                break
            prev_micro = micro_score
            sys.stdout.flush()
        n+=1
    print('Average Training Loss',sum(losses)/len(losses))
    #'''
    precision = tp/(tp+fp)
    recall = tp/total_mesh
    if precision+recall==0:
        print('Micro score at 0')
    else:
        micro_score = 2*precision*recall/(precision+recall)
        print('Training set micro score',micro_score, '| Recall',recall,'| Precision',precision)
        
    ground_recall = g_tp/total_mesh
    ground_precision = g_tp/(g_tp+g_fp)
    ground_micro = 2*ground_recall*ground_precision/(ground_recall+ground_precision)
    print('GROUNDTRUTH f-score',ground_micro,'| Recall',ground_recall,'| Precision',ground_precision)
    #'''
    
    if early_stop:
        break
    print('------------------End of Epoch',e,'-------------------')

    
