import numpy as np
import json
import torch
import random
from random import shuffle
from collections import Counter
import argparse
import time

########################################################################
# Basic Word2Vec - CBOW and Skip Gram
#
# 처리 순서 1
#  ㄴ text8 코퍼스에서 단어 토큰화 및 단어에 ID값 부여
#  ㄴ 기존에 미리 코퍼스에서 단어 토큰화 및 단어에 ID값 부여했다면
#     npy 파일과 json 파일에서 불러오기
#
# 처리 순서 2
#  ㄴ CBOW 모델과 Skip-gram 모델 중 선택하여 단어 임베딩 학습
#  ㄴ 추후 W_emb, W_out를 npy 파일로 저장해서, 다시 학습할 일이 없도록 만들 수 있음
#
# 처리 순서 3
#  ㄴ 학습 완료된 W_emb, W_out을 가지고 유사도 검사 수행
########################################################################

########################################################################
# 이정주 학우님 코드.. 살짝 debug 방식으로 바꿨음.
# preprocess : 코퍼스 내 단어 토큰화 및 단어에 ID 값 부여
# <input>
#  corpus : 코퍼스(말뭉치 파일). text8 파일을 사용.
#  is_debug : 디버그 모드 설정. 디버그 모드일 경우, 코퍼스 맨 앞에서 1000개 단어만 수집
#
# <output>
#  corpus : 단어가 토큰화를 완료한 코퍼스
#  word_to_id : 단어(key)에 ID(value)를 매핑한 dictionary
#
########################################################################
def preprocess(corpus, is_debug=False):
    word_to_id = {}
    id_to_word = {}

    corpus = corpus.lower()
    corpus = corpus.replace('.', ' .')
    words = corpus.split(' ')

    if is_debug:
        words = words[:2000]

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word

########################################################################
# read_corpus : 지정된 경로에서 텍스트 파일 읽어오기
# <input>
#  file_path : 지정된 파일의 경로
#
# <output>
#  corpus : 지정한 파일에서 읽어들인 코퍼스
#
########################################################################
def read_corpus(file_path):
    f = open(file_path, 'r')
    corpus = f.readline()
    f.close()

    return corpus

########################################################################
# Skipgram : Skipgram 모델 구현
# <input>
#  center : 중심 단어 index
#  context : 주변 단어 index
#  inputMatrix : 초기 W_emb
#  outputMatrix : 초기 W_out
#
#  주의할 점은, inputMatrix와 outputMatrix의 형태가 (V, D)로 동일.
#  중간에 transpose를 취하거나, 행렬-벡터 곱 연산 수행시 조심해야 함.
#
# <output>
#  loss : Loss Value. cross entropy로 구함
#  grad_emb : W_emb에 대한 gradient
#  grad_out : W_out에 대한 gradient
#
########################################################################
def Skipgram(center, context, inputMatrix, outputMatrix):

    ####################################################################
    # inputMatrix에서 center word에 해당하는 index에 있는 부분만을 뽑아낸다.
    # h.shape = [64]
    ####################################################################
    h = inputMatrix[center]

    ####################################################################
    # outputMatrix와 h를 곱해서 ouput를 얻어내기
    #
    # outputMatrix는 (V, 64)
    # h.view(-1, 1)는 (64, 1)
    # o는 (V, 64) * (64, 1) = (V, 1)
    ####################################################################
    o = torch.mm(outputMatrix, h.view(-1, 1))

    ####################################################################
    # outputMatrix와 h를 곱해서 ouput를 얻어내기
    #
    # outputMatrix는 (V, 64)
    # h.view(-1, 1)는 (64, 1)
    # o는 (V, 64) * (64, 1) = (V, 1)
    ####################################################################
    y = torch.nn.functional.softmax(o, dim=0)
    g = y.clone()
    g[context] -= 1

    ####################################################################
    # loss value는 단순히 cross entropy를 이용해서 얻는다.
    ####################################################################
    loss = -torch.log(y[context])

    ####################################################################
    # grad_out를 계산
    # 
    # dL/dW_out = h * g'
    # 하지만, 공식과 다르게, W_out는 (V, 64) matrix
    # 따라서, 코드의 W_out의 형태를 맞추면서, 올바르게 게산하기 위해선, 
    # h * g'이 아니라, g * h'으로 계산해야 함.
    # g는 (V, 1)의 vector
    # h'은 (1, 64)의 vector. h.view(1, -1)로 간단히 transpose
    #
    # 따라서, grad_out은 (V, 64)의 matrix가 됨
    #
    ####################################################################
    grad_out = torch.mm(g, h.view(1, -1))
    
    ####################################################################
    # grad_emb를 계산
    # 
    # dL/dh = W_out * g
    # 하지만, 공식과 다르게, W_out는 (V, 64) matrix
    # 그리고 g는 (V, 1) vector
    # 따라서, 현재 코드에서 W_out은 이미 transpose된 상태라고 가정한다면
    # g도 transpose를 해야 함. 간단하게 view(1, -1)을 통해 
    # (1, V) vector로 변환해서 계산
    #
    # grad_emb는 (1, 64)의 행 vector를 얻게 됨.
    #
    ####################################################################
    grad_emb = torch.mm(g.view(1, -1), outputMatrix)

    return loss, grad_emb, grad_out

########################################################################
# CBOW : CBOW 모델 구현
# <input>
#  center : 중심 단어 index
#  context : 주변 단어 index list
#  inputMatrix : 초기 W_emb
#  outputMatrix : 초기 W_out
#
#  주의할 점은, inputMatrix와 outputMatrix의 형태가 (V, D)로 동일.
#  중간에 transpose를 취하거나, 행렬-벡터 곱 연산 수행시 조심해야 함.
#
# <output>
#  loss : Loss Value. cross entropy로 구함
#  grad_emb : W_emb에 대한 gradient
#  grad_out : W_out에 대한 gradient
#
########################################################################
def CBOW(center, context, inputMatrix, outputMatrix):

    ####################################################################
    # inputMatrix에서 context list에 해당하는
    # index에 있는 부분만을 뽑아, torch sum을 수직 방향으로 더하기.
    # h.shape = [64]
    ####################################################################
    h = torch.sum(inputMatrix[context], dim=0)  # forward h.shape = [64]

    ####################################################################
    # outputMatrix와 h를 곱해서 ouput를 얻어내기
    #
    # outputMatrix는 (V, 64)
    # h.reshape(-1, 1)는 (64, 1)
    # o는 (V, 64) * (64, 1) = (V, 1)
    # 
    # y는 o에서 softmax를 취해 얻어낸다.
    # softmax cross entropy를 사용하기 때문에, target index에서 1을 빼준다.
    ####################################################################
    o = torch.mm(outputMatrix, h.reshape(-1, 1))
    y = torch.nn.functional.softmax(o, dim=0)
    g = y.clone()
    g[center] -= 1  # target index y에서 1을 뺀 값.= (V, 1)

    ####################################################################
    # loss value는 단순히 cross entropy에서 target 부분만 계산해서 얻는다.
    ####################################################################
    loss = -torch.log(y[center])

    ####################################################################
    # grad_emb를 계산
    # 
    # dL/dh = W_out * g
    # 하지만, 수업 시간에 배웠던 공식과 다르게, W_out는 (V, 64) matrix
    # 그리고 g는 (V, 1) vector
    # 따라서, 현재 코드에서 W_out은 이미 transpose된 상태라고 가정한다면
    # g도 transpose를 해야 함. 간단하게 reshape(1, -1)을 통해 
    # (1, V) vector로 변환해서 계산
    #
    # grad_emb는 (1, 64)의 행 vector를 얻게 됨.
    #
    ####################################################################
    grad_emb = torch.mm(g.reshape(1, -1), outputMatrix)
    
    ####################################################################
    # grad_out를 계산
    # 
    # dL/dW_out = h * g'
    # 하지만, 공식과 다르게, W_out는 (V, 64) matrix
    # 따라서, 코드의 W_out의 형태를 맞추면서, 올바르게 게산하기 위해선, 
    # h * g'이 아니라, g * h'으로 계산해야 함.
    # g는 (V, 1)의 vector
    # h'은 (1, 64)의 vector. h.reshape(1, -1)로 간단히 transpose
    #
    # 따라서, grad_out은 (V, 64)의 matrix가 됨
    #
    ####################################################################
    grad_out = torch.mm(g.reshape(-1,1), h.reshape(1,-1))

    return loss, grad_emb, grad_out

########################################################################
# getRandomContext : SGD 방식으로 학습을 진행할 때, training data set에서 
#                    임의로 training data를 선택
# <input>
#  corpus : 단어 토큰화가 완료된 코퍼스
#  C : window_size
#
# <output>
#  centerword : 중심 단어
#  context : 주변 단어 list 
#
########################################################################
def getRandomContext(corpus, C=5):
    wordID = random.randint(0, len(corpus) - 1)
    
    context = corpus[max(0, wordID - C):wordID] # left context words
    if wordID+1 < len(corpus):
        if type(corpus) is list:
            context += corpus[wordID+1:min(len(corpus), wordID + C + 1)] # right context words
        else:
            context = np.concatenate((context, corpus[wordID+1:min(len(corpus), wordID + C + 1)])) # right context words
    
    centerword = corpus[wordID]
    context = [w for w in context if w != centerword]
    
    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)

def word2vec_trainer(corpus, word2ind, mode="CBOW", dimension=64, learning_rate=0.05, iteration=10000):
                           # word2ind == word_to_id   

	  #initialization
    W_emb = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    W_out = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    window_size = 5

    losses=[]
    for i in range(iteration + 1):
        #Training word2vec using SGD
        centerWord, contextWords = getRandomContext(corpus, window_size)
        centerInd = centerWord # word2ind[centerWord]
        contextInds = contextWords # [word2ind[w] for w in contextWords]

        #learning rate decay
        lr = learning_rate*(1-i/iteration)

        if mode == "CBOW":
            L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out)
            W_emb[contextInds] -= lr*G_emb
            W_out -= lr*G_out
            losses.append(L.item())

        elif mode == "SG":
            for contextInd in contextInds:
                L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out)
                W_emb[centerInd] -= lr*G_emb.squeeze()
                W_out -= lr*G_out
                losses.append(L.item())
        else:
            print("Unkwnown mode : "+mode)
            exit()

        if i%1000==0:
            avg_loss=sum(losses)/len(losses)
            print("%d Loss : %f" %(i / 1000,  avg_loss))
            losses=[]

    return W_emb, W_out

def Analogical_Task(word_to_id, id_to_word, word_matrix):
    ART = [
           ['do', 'did', 'have', 'had'],
           ['brother', 'sister', 'grandson', 'granddaughter'],
           ['apparent', 'apparently', 'rapid', 'rapidly'],
           ['possibly', 'impossibly', 'ethical', 'unethical'],
           ['great', 'greater', 'tough', 'tougher'],
           ['easy', 'easiest', 'lucky', 'luckiest'],
           ['think', 'thinking', 'read', 'reading'],
           ['walking', 'walked', 'swimming', 'swam'],
           ['mouse', 'mice', 'dollar', 'dollars'],
           ['work', 'works', 'speak', 'speaks']
           ]

    print("querys are assinged")

    problem_vec = []
    for relation in ART:
        idx_1 = word_to_id[relation[0]]
        idx_2 = word_to_id[relation[1]]
        idx_3 = word_to_id[relation[2]]
        idx_4 = word_to_id[relation[3]]

        problem_vec.append([relation[3], word_matrix[idx_2] - word_matrix[idx_1] + word_matrix[idx_3]])
        problem_vec.append([relation[2], word_matrix[idx_1] - word_matrix[idx_2] + word_matrix[idx_4]])
        problem_vec.append([relation[1], word_matrix[idx_4] - word_matrix[idx_3] + word_matrix[idx_1]])
        problem_vec.append([relation[0], word_matrix[idx_3] - word_matrix[idx_4] + word_matrix[idx_2]])

        # problem_vec.append([relation[3], word_matrix[word_to_id[relation[3]]]])
        # problem_vec.append([relation[2], word_matrix[word_to_id[relation[2]]]])
        # problem_vec.append([relation[1], word_matrix[word_to_id[relation[1]]]])
        # problem_vec.append([relation[0], word_matrix[word_to_id[relation[0]]]])
        
    print("number of questions ( 36 ) ", len(problem_vec))
    print("Task START")
    for question in problem_vec:
        most_similar(question, word_to_id, id_to_word, word_matrix)
    print("DONE_TASK")


def cos_similarity(x, y, eps=1e-8):
    x_np = x.numpy()
    y_np = y.numpy()
    nx = x_np / np.linalg.norm(x_np)
    ny = y_np / np.linalg.norm(y_np)

    return np.dot(nx, ny)
    


def most_similar(query, word_to_id, id_to_word, word_matrix, top=10):
    if query[0] not in word_to_id:
        print('%s (을)를 찾을 수 없습니다.' % query)
        return

    print('\n[answer is] ' + query[0])
    answer_id=word_to_id[query[0]] 
    answer_vec = word_matrix[answer_id]
    
    query_vec = query[1]

    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if type(list(id_to_word.keys())[0]) == str:  
            print(id_to_word[str(i)]) # from loaded file
        else:
            print(id_to_word[i])      # just after training
        count += 1
        if count >= top:
            return

def main(load_saved=False, is_debug=False, mode = "CBOW"):
    # Write your code of data processing, training, and evaluation
	  # Full training takes very long time. We recommend using a subset of text8 when you debug
    corpus = None
    word_to_id = None
    id_to_word = None
    W_emb = None
    W_out = None

    start_time = time.time()

    print('Start preprocessing')

    if not load_saved:
        corpus = read_corpus('text8')
        corpus, word_to_id, id_to_word = preprocess(corpus, is_debug=is_debug)
        
        W_emb, W_out = word2vec_trainer(corpus, word_to_id, learning_rate=0.00001, mode=mode)

        np.save('preprocess_corpus.npy', corpus)

        torch.save(W_emb, 'W_emb.pt')
        torch.save(W_out, 'W_out.pt')

        with open('preprocess_word_to_id.json', 'w') as f:
            json.dump(word_to_id, f)
        with open('preprocess_id_to_word.json', 'w') as f:
            json.dump(id_to_word, f)
    else:
        corpus = np.load('preprocess_corpus.npy')

        W_emb = torch.load('W_emb.pt')
        W_out = torch.load('W_out.pt')

        with open('preprocess_word_to_id.json', 'r') as f:
            word_to_id = json.load(f)
        with open('preprocess_id_to_word.json', 'r') as f:
            id_to_word = json.load(f)
    
    print('Finished preprocessing')

    print("--- %s minutes ---" % ((time.time() - start_time) / 60))

    Analogical_Task(word_to_id, id_to_word, W_emb)
    

main(load_saved=False, is_debug=True, mode="SG") 