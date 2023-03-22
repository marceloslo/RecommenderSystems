
import numpy as np
import pandas as pd

class SVD:
    #cria modelo de funksvd com os parametros
    def __init__(self,k=25,alpha=0.005,regularization=0.02,epochs=20,biased=False,seed=None):
        np.random.seed(seed)
        self.k=k
        self.alpha = alpha
        self.regularization = regularization
        self.epochs = epochs
        self.seed = seed
        self.biased = biased

    #cria um mapeamento rapidamente de ids para itens, permitindo utilizar indices de 0 a n na matriz sem problemas
    def idsToIndex(self,df):
        itemConverter = {}
        userConverter = {}
        userIndex=0
        itemIndex=0
        for ids in df['UserId:ItemId']:
            ids = ids.split(':')
            userId = ids[0]
            itemId = ids[1]
            #se usuário não tiver um indice
            conversion = userConverter.get(userId,-1)
            if conversion == -1:
                #cria um novo indice
                userConverter[userId] = userIndex
                userIndex +=1
            #se item não tiver um indice
            conversion = itemConverter.get(itemId,-1)
            if conversion == -1 :
                #cria um novo indice
                itemConverter[itemId] = itemIndex
                itemIndex +=1
        #retorna mapeamentos
        return userConverter,itemConverter

        
    #inicializa todas matrizes necessárias (incluindo o proprio dataframe que é embaralhado para evitar bias)
    def initialize(self,df):
        self.S = df.sample(frac=1).reset_index(drop=True)
        self.converters = self.idsToIndex(self.S)
        #média global dos itens do dataframe
        self.mean = np.mean(self.S['Rating'])
        self.S = self.S.to_dict(orient='records')
        self.m = len(self.converters[0])
        self.n = len(self.converters[1])
        self.P = np.random.normal(0,0.1,size=(self.m, self.k))
        self.Q = np.random.normal(0,0.1,size=(self.k, self.n))
        #Bias de usuário e item respectivamente
        self.bu = np.zeros(self.m)
        self.bi = np.zeros(self.n)

    # treina o modelo no dataframe fornecido
    def fit(self,df):
        self.initialize(df)
        for epoch in range(self.epochs):
            for row in self.S:

                #formata os dados da linha do dataframe
                r = row['Rating']
                ids = row['UserId:ItemId'].split(':')
                u = self.converters[0][ids[0]]
                i = self.converters[1][ids[1]]

                r_hat = self.mean+np.dot(self.P[u,:],self.Q[:,i])
                #calcula previsão e erro
                if self.biased:
                    r_hat += self.bu[u]+self.bi[i]
                error = r - r_hat

                #atualiza as matrizes P e Q e os user e item biases (bu,bi)
                if self.biased:
                    self.bu[u] = self.bu[u] + self.alpha*(error - self.regularization*self.bu[u])
                    self.bi[i] = self.bi[i] + self.alpha*(error - self.regularization*self.bi[i])
                p_u = self.P[u,:] + self.alpha*(error*self.Q[:,i] - self.regularization*self.P[u,:])
                q_i = self.Q[:,i] + self.alpha*(error*self.P[u,:] - self.regularization*self.Q[:,i])
                self.P[u,:] = p_u
                self.Q[:,i] = q_i
            
    #preve o rating de um único item
    def predict(self,user,item):
        r_hat = self.mean+np.dot(self.P[user,:],self.Q[:,item])
        if self.biased:
            r_hat += self.bu[user]+self.bi[item]
        return r_hat
    
    #prevê e formata um dataframe como pedido no requisito do trabalho
    def predictTargets(self,targets):
        predictions = []
        for id in targets['UserId:ItemId']:
            ids = id.split(':')
            #caso o item ou usuário não tenha sido visto no treino, usa a média como previsão
            if (self.converters[0].get(ids[0],-1) == -1) or (self.converters[1].get(ids[1],-1)==-1):
                predictions.append([id,self.mean])
                continue
            user = self.converters[0][ids[0]]
            item = self.converters[1][ids[1]]
            #caso contrário, faz a previsão
            r_hat = self.predict(user,item)
            #limita nos valores possíveis
            r_hat = r_hat if r_hat < 5 else 5
            r_hat = r_hat if r_hat > 1 else 1
            predictions.append([id,r_hat])
        return predictions

    #avalia o erro do modelo em relação a algum conjunto de dados
    def evaluate(self,df,train=False):
        error=0
        X = df['UserId:ItemId']
        y = df['Rating']
        for i in range(len(X)):
            ids = X[i].split(':')
            #caso o item ou usuário não tenha sido visto no treino, usa a média como previsão
            if (self.converters[0].get(ids[0],-1) == -1) or (self.converters[1].get(ids[1],-1)==-1):
                error += (y[i] - self.mean)**2
                continue
            user = self.converters[0][ids[0]]
            item = self.converters[1][ids[1]]
            r_hat = self.predict(user,item)
            if not train:
                r_hat = r_hat if r_hat < 5 else 5
                r_hat = r_hat if r_hat > 1 else 1
            error += (y[i] - r_hat)**2
        return np.sqrt(error/len(df))