import numpy as np

class KNN:
  

  def __init__(self, k):
    self.k = k


  #training
  def fit(self, X, Y):
    self.X_train = X
    self.Y_train = Y


  def oghlidosi(self, x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


  def predict (self, X): #x bozorg yaani listi az data va x kochak yaani yek data
    Y = []
    for x in X:
      distances = []
      for x_train in self.X_train:
        d= self.oghlidosi(x, x_train)
        distances.append(d)

      nearest_neighbors = np.argsort(distances)[0 : self.k]
      result = np.bincount(self.Y_train[nearest_neighbors])
      y = np.argmax(result)
      Y.append(y)
    return Y


  def evaluate(self, X, Y):
    Y_pred = self.predict(X)
    acuuracy = np.sum(Y_pred == Y)/ len(Y)
    return acuuracy