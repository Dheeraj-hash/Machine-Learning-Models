import numpy as np
import pickle as pkl

class LDA:
    def __init__(self,k):
        self.n_components = k
        self.linear_discriminants = None

    def fit(self, X, y):
        """
        X: (n,d,d) array consisting of input features
        y: (n,) array consisting of labels
        return: Linear Discriminant np.array of size (d*d,k)
        """
        # TODO
        self.linear_discriminants=np.zeros((len(X[0]*len(X[0])),self.n_components))
        X=X.reshape(X.shape[0],-1)
        d = X.shape[1]*X.shape[2]
        m=np.mean(X,axis=0)
        Sb = np.zeros((d, d))
        Sw = np.zeros((d, d))
        
        for i in np.unique(y):
            new_X=X[y == i]
            mean=np.mean(new_X,axis=0)
            mean_diff = (mean - m)
            mean_diff.reshape(-1,1)
            Sb += new_X.shape[0] * mean_diff.dot(mean_diff.T)
            Sw += (new_X - mean).T.dot(new_X - mean)
        S=np.linalg.pinv(Sw).dot(Sb)
        eig_values, eig_vectors = np.linalg.eig(S)
        eig_pairs = []
        for i in range(len(eig_values)):
            eig_pairs.append((np.abs(eig_values[i]), eig_vectors[:,i]))
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        linear_discriminants = []
        for i in range(self.n_components):
            linear_discriminants.append(eig_pairs[i][1].reshape(d, 1))
        self.linear_discriminants = np.hstack(linear_discriminants)
        return self.linear_discriminants      
        # Modify as required
        #END TODO 
    
    def transform(self, X, w):
        """
        w:Linear Discriminant array of size (d*d,1)
        return: np-array of the projected features of size (n,k)
        """
        # TODO
         # Modify as required
        X=X.reshape(X.shape[0],-1)
        projection=np.matmul(X,w)
        return projection                # Modify as required
        # END TODO


# if _name_ == 'main':
    # mnist_data = 'mnist.pkl'
    # with open(mnist_data, 'rb') as f:
    #     data = pkl.load(f)
# print("hi")
# X=np.random.rand(4,2,2)
# y=np.array([0,0,1,1])
# k=2
# lda = LDA(k)
# w=lda.fit(X, y)
# X_lda = lda.transform(X,w)
# print(X_lda)