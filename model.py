import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.gaussian = {}
        self.bernoulli = {}
        self.laplace = {}
        self.exponential = {}
        self.multinomial = {}
    
    def fit(self, X, y):

        """Start of your code."""
        """
        X : np.array of shape (n,10)
        y : np.array of shape (n,)
        Create a variable to store number of unique classes in the dataset.
        Assume Prior for each class to be ratio of number of data points in that class to total number of data points.
        Fit a distribution for each feature for each class.
        Store the parameters of the distribution in suitable data structure, for example you could create a class for each distribution and store the parameters in the class object.
        You can create a separate function for fitting each distribution in its and call it here.
        """
        frequency_dict = {}
        data_by_label = {}
        y=y.reshape(-1,1)
        data = np.hstack((X,y))
        
        for row in data:
            label = row[10]
            if label in data_by_label:
                data_by_label[label].append(row)
            else:
                data_by_label[label] = [row]

        for label in data_by_label:
            q=data_by_label[label]
            qq=np.array(q,dtype='float32')
            q1=np.array(qq,dtype='float32')
            
            frequency_dict[label]=len(q1)
            
            mean_x1,mean_x2 = np.mean(q1[:,0]), np.mean(q1[:,1])
            var_x1,var_x2 = np.var(q1[:,0]), np.var(q1[:,1])
            self.gaussian[label]=[mean_x1,mean_x2,var_x1,var_x2]
            
            p_x3,p_x4=np.sum(q1[:,2])/len(q1),np.sum(q1[:,3])/len(q1)
            self.bernoulli[label]=[p_x3,p_x4]
            
            mu_x5,mu_x6 = np.median(q1[:,4]), np.median(q1[:,5])
            b_x5,b_x6 = np.mean(np.abs(q1[:,4] - mu_x5)) , np.mean(np.abs(q1[:,5] - mu_x6))
            self.laplace[label]=[mu_x5,mu_x6,b_x5,b_x6]
            
            lambda_x7,lambda_x8 = 1/np.mean(q1[:,6]), 1/np.mean(q1[:,7])
            self.exponential[label] = [lambda_x7,lambda_x8]
            
            total_count = len(q1)
            count1 = {0:0,1:0,2:0,3:0,4:0}

            for num in q1[:,8]:
                if num in count1:
                    count1[num] += 1
                else:
                    count1[num] = 1
            
            count=list(count1.values())
            zz=[i / total_count for i in count]
            [p0_x9,p1_x9,p2_x9,p3_x9,p4_x9] = zz
            
            count1 = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}

            for num in q1[:,9]:
                if num in count1:
                    count1[num] += 1
                else:
                    count1[num] = 1
            
            count=list(count1.values())
            zz=[i / total_count for i in count]
            [p0_x10,p1_x10,p2_x10,p3_x10,p4_x10,p5_x10,p6_x10,p7_x10] = zz
            self.multinomial[label] = [[p0_x9,p1_x9,p2_x9,p3_x9,p4_x9],[p0_x10,p1_x10,p2_x10,p3_x10,p4_x10,p5_x10,p6_x10,p7_x10]]
        priors = frequency_dict
        constant = sum(frequency_dict.values())

        # Divide every value in the dictionary by the constant
        for key in frequency_dict:
            priors[key] /= constant
        self.priors = priors
        """End of your code."""
        
    def log_gaussian(self,x,mu,var):
        sigma=np.math.sqrt(var)
        log_pdf = -0.5 * ((x - mu) / sigma) ** 2 - np.math.log(sigma)
        return log_pdf
    
    def log_bernoulli(self,x,p):
        log_pdf=x * np.math.log(p) + (1 - x) * np.math.log(1 - p)
        return log_pdf
    
    def log_laplace(self,x,mu,b):
        log_pdf =  - np.abs(x - mu) / b  - np.math.log(2*b)
        return log_pdf
    
    def log_exponential(self,x,lambda1):
        log_pdf = - lambda1 * x + np.math.log(lambda1) 
        return log_pdf
    
    def log_multinomial(self,x,p):
        log_pdf=np.zeros(x.shape)
        for i in range(x.shape[0]):
            log_pdf[i] = np.math.log(p[int(x[i])])
        return log_pdf

    def predict(self, X):
        
        """Start of your code."""
        """
        X : np.array of shape (n,10)

        Calculate the posterior probability using the parameters of the distribution calculated in fit function.
        Take care of underflow errors suitably (Hint: Take log of probabilities)
        Return an np.array() of predictions where predictions[i] is the predicted class for ith data point in X.
        It is implied that prediction[i] is the class that maximizes posterior probability for ith data point in X.
        You can create a separate function for calculating posterior probability and call it here.
        """
        score=np.zeros((len(X),3))
        for i in range(3):
            z=np.zeros(X.shape)
            z[:,0]= self.log_gaussian(X[:,0],self.gaussian[i][0],self.gaussian[i][2])
            z[:,1]= self.log_gaussian(X[:,1],self.gaussian[i][1],self.gaussian[i][3])
            z[:,2]= self.log_bernoulli(X[:,2],self.bernoulli[i][0])
            z[:,3]= self.log_bernoulli(X[:,3],self.bernoulli[i][1])
            z[:,4]= self.log_laplace(X[:,4],self.laplace[i][0],self.laplace[i][2])
            z[:,5]= self.log_laplace(X[:,5],self.laplace[i][1],self.laplace[i][3])
            z[:,6]= self.log_exponential(X[:,6],self.exponential[i][0])
            z[:,7]= self.log_exponential(X[:,7],self.exponential[i][1])
            z[:,8]= self.log_multinomial(X[:,8],self.multinomial[i][0])
            z[:,9]= self.log_multinomial(X[:,9],self.multinomial[i][1])
            score[:,i]= self.priors[i]+np.sum(z,axis=1)
        pred=np.argmax(score,axis=1)
        return pred
        """End of your code."""

    def getParams(self):
        """
        Return your calculated priors and parameters for all the classes in the form of dictionary that will be used for evaluation
        Please don't change the dictionary names
        Here is what the output would look like:
        priors = {"0":0.2,"1":0.3,"2":0.5}
        c_gaussian = {"0":[mean_x1,mean_x2,var_x1,var_x2],"1":[mean_x1,mean_x2,var_x1,var_x2],"2":[mean_x1,mean_x2,var_x1,var_x2]}
        c_bernoulli = {"0":[p_x3,p_x4],"1":[p_x3,p_x4],"2":[p_x3,p_x4]}
        c_laplace = {"0":[mu_x5,mu_x6,b_x5,b_x6],"1":[mu_x5,mu_x6,b_x5,b_x6],"2":[mu_x5,mu_x6,b_x5,b_x6]}
        c_exponential = {"0":[lambda_x7,lambda_x8],"1":[lambda_x7,lambda_x8],"2":[lambda_x7,lambda_x8]}
        c_multinomial = {"0":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"1":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"2":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]]}
        
        X1	X2	: gaussian : mean, variance :f(x | μ, σ²) = (1 / √(2πσ²)) * exp(-((x - μ)²) / (2σ²))
        X3	X4	: bernoulli: probablity of class x=1 : P(X = x) = p^x * (1 - p)^(1 - x)
        X5	X6	: laplace : f(x | μ, b) = (1 / 2b) * exp(-|x - μ| / b)
        X7	X8	: exponential : f(x | λ) = λ * exp(-λx)
        X9	X10 : multinomial : P(X_1 = x_1, X_2 = x_2, ..., X_k = x_k) = (n! / (x_1! * x_2! * ... * x_k!)) * (p_1^x_1 * p_2^x_2 * ... * p_k^x_k)
        """
        priors = {}
        gaussian = {}
        bernoulli = {}
        laplace = {}
        exponential = {}
        multinomial = {}

        """Start your code"""
        priors = self.priors
        gaussian = self.gaussian
        bernoulli = self.bernoulli
        laplace = self.laplace
        exponential = self.exponential
        multinomial = self.multinomial
        
        """End your code"""
        return (priors, gaussian, bernoulli, laplace, exponential, multinomial)        

def save_model(model,filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open("model.pkl","wb")
    pkl.dump(model,file)
    file.close()

def load_model(filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename,"rb")
    model = pkl.load(file)
    file.close()
    return model

def visualise(data_points,labels):
    """
    datapoints: np.array of shape (n,2)
    labels: np.array of shape (n,)
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Generated 2D Data from 5 Gaussian Distributions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def net_f1score(predictions, true_labels):
    """Calculate the multclass f1 score of the predictions.
    For this, we calculate the f1-score for each class 

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float(list): The f1 score of the predictions for each class
    """

    def precision(predictions, true_labels, label):
        """Calculate the multclass precision of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The precision of the predictions.
        """
        """Start of your code."""
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        for predicted, true in zip(predictions, true_labels):
            if predicted == true:
                if label == predicted:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if label == predicted:
                    false_positives += 1
                else:
                    false_negatives += 1       
        
        if true_positives + false_positives == 0:
            return 0.0  # To handle the case when there are no positive predictions.

        P = true_positives / (true_positives + false_positives)
        return P

        """End of your code."""
        
    def recall(predictions, true_labels, label):
        """Calculate the multclass recall of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The recall of the predictions.
        """
        """Start of your code."""
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        for predicted, true in zip(predictions, true_labels):
            if predicted == true:
                if label == predicted:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if label == predicted:
                    false_positives += 1
                else:
                    false_negatives += 1       
        
        if true_positives + false_negatives == 0:
            return 0.0  # To handle the case when there are no positive predictions.

        R = true_positives / (true_positives + false_negatives)
        return R
        """End of your code."""
        
    def f1score(predictions, true_labels, label):
        """Calculate the f1 score using it's relation with precision and recall.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The f1 score of the predictions.
        """

        """Start of your code."""
        p=precision(predictions, true_labels, label)
        r=recall(predictions, true_labels, label)
        f1=2*p*r/(p+r)
        """End of your code."""
        return f1
    
    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s

def accuracy(predictions,true_labels):
    """

    You are not required to modify this part of the code.

    """
    return np.sum(predictions==true_labels)/predictions.size

if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./data/train_dataset.csv',index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./data/validation_dataset.csv',index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:,:-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    # visualise(train_datapoints, train_labels)

    # Train the model
    model = NaiveBayes()
    model.fit(train_datapoints, train_labels)

    # Make predictions
    train_predictions = model.predict(train_datapoints)
    validation_predictions = model.predict(validation_datapoints)

    # Calculate the accuracy
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, validation_labels)

    # Calculate the f1 score
    train_f1score = net_f1score(train_predictions, train_labels)
    validation_f1score = net_f1score(validation_predictions, validation_labels)

    # Print the results
    print('Training Accuracy: ', train_accuracy)
    print('Validation Accuracy: ', validation_accuracy)
    print('Training F1 Score: ', train_f1score)
    print('Validation F1 Score: ', validation_f1score)

    # Save the model
    save_model(model)

    # Visualize the predictions
    #visualise(validation_datapoints, validation_predictions)#, "validation_predictions.png")