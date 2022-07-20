from itertools import count
from game import FlappyBird

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

class Transition:
    def __init__(self, state, action, nextState, reward):
        self.state = state
        self.action = action
        self.nextState = nextState
        self.reward = reward
    
    def __iter__(self):
        for i in range(4):
            if i == 0:
                yield self.state 
            elif i == 1:
                yield self.action
            elif i == 2:
                yield self.nextState
            elif i == 3:
                yield self.reward

class Memory:
    def __init__(self, maxCapacity=10000):

        self.maxCapacity = maxCapacity
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

        if len(self.memory) > self.maxCapacity:
            self.memory.pop(0)

    def sample(self, batchSize):
        return random.sample(self.memory, batchSize)

class Model(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(Model, self).__init__()

        self.lin1 = nn.Linear(inputSize, 100).float()
        self.act1 = nn.ReLU().float()
        self.lin2 = nn.Linear(100, 100).float()
        self.act2 = nn.ReLU().float()
        self.lin3 = nn.Linear(100, outputSize).float()
        self.act3 = nn.Softmax().float()

    def forward(self, input):
        input = self.lin1(input)
        input = self.act1(input)
        input = self.lin2(input)
        input = self.act2(input)
        input = self.lin3(input)
        output = self.act3(input)

        return output

class Agent:
    def __init__(self, stateLen, numActions, learningRate=0.01, gamma=0.99, epsilonStart=0.9, epsilonEnd=0.05, epsilonDecay=200, targetUpdate=10, batchSize=100, maxMemoryCapacity=10000):
        self.lr = learningRate
        self.gamma = gamma
        self.batchSize = batchSize
        self.epsilonStart = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.epsilonDecay = epsilonDecay
        self.targetUpdate = targetUpdate

        self.device = "cpu"

        self.policy = Model(stateLen, numActions).to(self.device)
        self.targetNet = Model(stateLen, numActions).to(self.device)
        self.targetNet.load_state_dict(self.policy.state_dict())
        self.targetNet.eval()

        self.optimizer = optim.RMSprop(self.policy.parameters())
        self.memory = Memory(maxCapacity=maxMemoryCapacity)
        self.game = FlappyBird()

        self.steps = 0
    
    def getAction(self, state):

        epsilonThreshold = self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * math.exp(-1 * self.steps / self.epsilonDecay)
        
        if random.random() > epsilonThreshold:
            with torch.no_grad():
                print("CALC MOVE")
                print(self.policy(state).max())
                return torch.tensor([[self.policy(state).max()]], device=self.device, dtype=torch.float)
        else:
            print("RANDOM MOVE")
            return torch.tensor([[random.randint(0, 1)]], device=self.device, dtype=torch.float)
        
    def plot(self):
        pass

    def optimizeModel(self):

        if len(self.memory.memory) < self.batchSize:
            return
        
        batch = self.memory.sample(self.batchSize)
        batch = Transition(*zip(*batch))

        tempMask = torch.tensor(tuple(map(lambda s: s is not None, batch.nextState)), device=self.device, dtype=torch.bool)
        tempNextStates = torch.stack([state for state in batch.nextState if state is not None])

        stateBatch = torch.stack(batch.state)
        actionBatch = torch.stack(batch.action)
        rewardBatch = torch.stack(batch.reward)

        stateActionValues = self.policy(stateBatch)

        nextStateValues = torch.zeros(self.batchSize, device=self.device)
        nextStateValues[tempMask] = self.targetNet(tempNextStates).max(1)[0].detach()

        print(nextStateValues * self.gamma)

        expectedStateActionValues = (nextStateValues * self.gamma) + rewardBatch

        #HuberLoss???? IDK how to calculate this but it is a loss function
        criterion = nn.SmoothL1Loss()
        loss = criterion(stateActionValues, expectedStateActionValues.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for parameter in self.policy.parameters():
            parameter.grad.data.clamp(-1, 1)
        
        self.optimizer.step()
    
    def train(self, episodes):
        for i in range(episodes):
            print(i)
            self.game.reset()

            state = self.game.getState()

            for t in count():
                action = self.getAction(state)
                reward, gameOver = self.game.gameStep(action)

                reward = torch.tensor([reward], device=self.device)

                newState = self.game.getState()

                self.memory.push(state, action, newState, reward)

                state = newState

                self.optimizeModel()
                if gameOver:
                    break

                if i % self.targetUpdate == 0:
                    self.targetNet.load_state_dict(self.policy.state_dict())
        
        print("Complete")

a = Agent(4, 2)
a.train(300)