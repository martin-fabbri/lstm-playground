import numpy as np

data = open('kafka.txt', 'r').read()

chars = list(set(data))

dataSize, vocabSize = len(data), len(chars)

print(f'data has {dataSize} chars, {vocabSize} unique')

charToIx = {ch: i for i, ch in enumerate(chars)}

ixToChar = {i: ch for i, ch in enumerate(chars)}

print(charToIx)
print(ixToChar)

vectorForCharA = np.zeros((vocabSize, 1))
vectorForCharA[charToIx['a']] = 1
print(vectorForCharA)

# hyperparameters

hiddenSize = 100
seqLenght = 25
learningRate = 1e-1

# model parameters
wxh = np.random.rand(hiddenSize, vocabSize) * 0.01
whh = np.random.rand(hiddenSize, hiddenSize) * 0.01
why = np.random.rand(vocabSize, hiddenSize) * 0.01

# bias
bh = np.zeros((hiddenSize, 1))
by = np.zeros((vocabSize, 1))
print(wxh)

