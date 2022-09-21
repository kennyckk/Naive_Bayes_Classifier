from sklearn.datasets import load_iris
import numpy as np

def loadData():
    #load the dataset from sklearn datasets
    iris = load_iris()
    X, y = iris["data"], iris['target']

    # to split the data set into train and test
    N, D = X.shape
    Ntrain = int(N * 0.8)
    np.random.seed(12)
    shuffle = np.random.permutation(N)  # shuffle is a list of randomized integer range from 0 to N
    xtrain = X[shuffle[:Ntrain]]
    ytrain = y[shuffle[:Ntrain]]
    xtest = X[shuffle[Ntrain:]]
    ytest = y[shuffle[Ntrain:]]

    return xtrain,ytrain, xtest,ytest


class NBC:
    def __init__(self, feature_types: list, num_classes: int):
        self.feature_types = feature_types
        self.num_classes = num_classes
        self.prior = {} # a dict to hold the prior distribution for each class
        self.feat_mean_n_std_per_class = {} # a dict to hold list of mean and std for each feat in each class
        for i in range(self.num_classes):
            self.feat_mean_n_std_per_class[i] = []
        self.p_y_given_x = None #to later initialize in self.Predict() to hold the posterior probability of each class

    def Fit(self, xtrain, ytrain):
        def get_Prior_Distribution(ytrain):
            N = len(ytrain)
            # to count for occurence of different classes over the train sample (Prior Distribution)
            for item in ytrain:
                if item not in self.prior:
                    self.prior[item] = 1 / N
                else:
                    self.prior[item] += 1 / N

        def get_Mean_Std_Per_Feature(xtrain, ytrain):

            for c in range(self.num_classes):#loop over each class to get mean and std for different features
                x_per_class = xtrain[ytrain == c]  # to get the data point belonging to this class

                for feat_num, feat in enumerate(
                        self.feature_types):  # loop for each feat and get their correpsonding mean & std
                    feat_mean = np.mean(x_per_class[:, feat_num])
                    feat_std = np.std(x_per_class[:, feat_num])
                    self.feat_mean_n_std_per_class[c].append((feat_mean, feat_std)) #mean& std stored in dict declared in class

        # update self.prior with prior distribution (πc.) per class
        get_Prior_Distribution(ytrain)
        # update self.feat_mean_n_std_per_class with mean and std for each class per feature to prepare later conditional distribution computing
        get_Mean_Std_Per_Feature(xtrain, ytrain)

    def Predict(self, xtest):

        def get_Condi_Dist(mean, std, x_per_feat):
            # plug in x per feature into Gaussian pdf to get the conditional distribution

            x_per_feat = np.array(x_per_feat)
            p_x_given_y_per_feat = (np.pi * std) * np.exp(-0.5 * ((x_per_feat - mean) / std) ** 2) #Gaussian PDF equation
            p_x_given_y_per_feat = p_x_given_y_per_feat.reshape(len(xtest), 1) #prepare for np array broadcasting

            return p_x_given_y_per_feat

        self.p_y_given_x = np.zeros((len(xtest), 0)) # initialize np array to hold for the posterior prob from each class

        for c in range(self.num_classes):#loop over each class to get condi prob for each features
            # to get back the mean and std array calculated from Training
            feat_mean_std = self.feat_mean_n_std_per_class[c]
            # to get the prior probability for each class and prepare to multiply them with condi prob from each feature
            condi_prob_per_class = np.array([[1] for _ in range(len(xtest))]) * np.log(self.prior[c])

            for feat_num, feat in enumerate(self.feature_types): #loop over each feature in that class to get condi prob
                mean, std = feat_mean_std[feat_num] #the idx of array denoting the class
                x_per_feat = xtest[:, feat_num]
                p_x_given_y_per_feat = get_Condi_Dist(mean, std, x_per_feat) # to get condi prob for each feature for all xtest points
                condi_prob_per_class += np.log(p_x_given_y_per_feat)
            self.p_y_given_x = np.column_stack([self.p_y_given_x, condi_prob_per_class]) #this will be a (30,3) shape holding posterior probability

        likelihood = np.argmax(self.p_y_given_x, axis=1) #get the class with largest posterior probability
        return likelihood

if __name__ == "__main__":
    xtrain, ytrain, xtest, ytest=loadData()
    nbc=NBC(feature_types=["r,r,r,r"], num_classes=3)
    nbc.Fit(xtrain, ytrain)
    yhat=nbc.Predict(xtest)
    accuracy=np.mean(yhat==ytest)
    print(accuracy)