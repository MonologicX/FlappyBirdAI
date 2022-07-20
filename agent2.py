import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

from game import FlappyBird

class DQN(nn.Module):
    def __init__(self, nInputs, nOutputs, learningRate):
        super(DQN, self).__init__()

        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.lr = learningRate

        self.layer1 = nn.Linear(self.nInputs, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, self.nOutputs)

        self.optimizer = optim.Adam(self.parameters(), self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)
    
    def forward(self, state):
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        actions = self.layer3(state)

        return actions

class Memory():
    def __init__(self, maxMemory, nStates):
        self.memCounter = 0
        self.maxMemory = maxMemory
        self.stateMemory = np.zeros((self.maxMemory, nStates), dtype=np.float32)
        self.nextStateMemory = np.zeros((self.maxMemory, nStates), dtype=np.float32)
        self.rewardMemory = np.zeros(self.maxMemory, dtype=np.float32)
        self.actionMemory = np.zeros(self.maxMemory, dtype=np.int32)
        self.gameOverMemory = np.zeros(self.maxMemory, dtype=np.bool)
    
    def push(self, state, nextState, reward, action, gameOver):
        i = self.memCounter % self.maxMemory

        self.stateMemory[i] = state
        self.nextStateMemory[i] = nextState
        self.rewardMemory[i] = reward
        self.actionMemory[i] = action
        self.gameOverMemory[i] = gameOver

        self.memCounter += 1

class Agent():
    def __init__(self, gamma, epsilonStart, learningRate, nStates, nActions, batchSize, maxMemory=100000, epsilonEnd=0.05, epsilonDecay=0.05):
        self.gamma = gamma
        self.epsilon = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.epsilonDecay = epsilonDecay
        self.lr = learningRate
        self.nStates = nStates
        self.nActions = nActions
        self.batchSize = batchSize

        self.actionSpace = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        self.DQN = DQN(self.nStates, self.nActions, self.lr)
        self.memory = Memory(maxMemory, self.nStates)

        self.game = FlappyBird()
    
    def getAction(self, state):
        if random.random() > self.epsilon:
            state = torch.tensor([state]).to(self.DQN.device)
            action = self.DQN.forward(state)
            action = torch.argmax(action).item()
        else:
            action = np.random.choice(self.actionSpace)
        
        return action
    
    def learn(self):
        if self.memory.memCounter < self.batchSize:
            print("INSUFFICIENT DATA")
            return
        
        self.DQN.optimizer.zero_grad()

        maxMemory = min(self.memory.memCounter, self.memory.maxMemory)
        batch = np.random.choice(maxMemory, self.batchSize, replace=False)

        batchIndex = np.arange(self.batchSize, dtype=np.float32)

        stateBatch = torch.tensor(self.memory.stateMemory[batch]).to(self.DQN.device)
        nextStateBatch = torch.tensor(self.memory.nextStateMemory[batch]).to(self.DQN.device)
        rewardBatch = torch.tensor(self.memory.rewardMemory[batch]).to(self.DQN.device)
        gameOverBatch = torch.tensor(self.memory.gameOverMemory[batch]).to(self.DQN.device)
        actionBatch = self.memory.actionMemory[batch]

        q = self.DQN.forward(stateBatch)[batchIndex, actionBatch]
        nextQ = self.DQN.forward(nextStateBatch)[batchIndex, actionBatch]

        nextQ[gameOverBatch] = 0.0

        target = rewardBatch + (self.gamma * torch.max(nextQ, dim=-1)[0])
        loss = self.DQN.loss(target, q).to(self.DQN.device)

        loss.backward()
        self.DQN.optimizer.step()

        if self.epsilon != self.epsilonEnd:
            self.epsilon -= self.epsilonDecay
            if self.epsilon < self.epsilonEnd:
                self.epsilon = self.epsilonEnd

    def plot(self, scores, epsilonHistory, avgScores):
        plt.clf()
        plt.title("Training...")

        plt.plot(scores, color='b')
        plt.plot(epsilonHistory, color='g')
        plt.plot(avgScores, color='r')

        plt.ylim(ymin=0)

        plt.show()
        
    def train(self, nGames):
        scores = []
        epsilonHistory = []
        avgScores = []

        for i in range(1, nGames + 1):
            gameOver = False
            while not gameOver:
                state = self.game.getState()

                action = self.getAction(state)

                reward, gameOver = self.game.trainStep(action)

                nextState = self.game.getState()

                self.memory.push(state, nextState, reward, action, gameOver)

                self.learn()

                print(f"Game: {i}, Reward: {reward}, Epsilon: {self.epsilon}, Score: {self.game.score}")

            scores.append(self.game.score)
            avgScores.append(sum(scores) / len(scores))
            epsilonHistory.append(self.epsilon)

            self.game.reset()
        
        self.plot(scores, epsilonHistory, avgScores)

a = Agent(0.9, 1, 0.03, 6, 2, 64, epsilonEnd=0.001, epsilonDecay=0.000015, maxMemory=100000)
a.train(1000)