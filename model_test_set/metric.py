import numpy as np
def subsetAccuracy(y_test, predictions):
    """
    The subset accuracy evaluates the fraction of correctly classified examples
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    subsetaccuracy : float
        Subset Accuracy of our model
    """
    subsetaccuracy = 0.0

    for i in range(y_test.shape[0]):
        same = True
        for j in range(y_test.shape[1]):
            if y_test[i,j] != predictions[i,j]:
                same = False
                break
        if same:
            subsetaccuracy += 1.0
    
    return subsetaccuracy/y_test.shape[0]


def hammingLoss(y_test, predictions):
    """
    The hamming loss evaluates the fraction of misclassified instance-label pairs
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    hammingloss : float
        Hamming Loss of our model
    """
    hammingloss = 0.0
    for i in range(y_test.shape[0]):
        aux = 0.0
        for j in range(y_test.shape[1]):
            if int(y_test[i,j]) != int(predictions[i,j]):
                aux = aux+1.0
        aux = aux/y_test.shape[1]
        hammingloss = hammingloss + aux
    
    return hammingloss/y_test.shape[0]

def accuracy(y_test, predictions):
    """
    Accuracy of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    accuracy : float
        Accuracy of our model
    """
    accuracy = 0.0

    for i in range(y_test.shape[0]):
        intersection = 0.0
        union = 0.0
        for j in range(y_test.shape[1]):
            if int(y_test[i,j]) == 1 or int(predictions[i,j]) == 1:
                union += 1
            if int(y_test[i,j]) == 1 and int(predictions[i,j]) == 1:
                intersection += 1
            
        if union != 0:
            accuracy = accuracy + float(intersection/union)

    accuracy = float(accuracy/y_test.shape[0])

    return accuracy



def precision(y_test, predictions):
    """
    Precision of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    precision : float
        Precision of our model
    """
    precision = 0.0

    for i in range(y_test.shape[0]):
        intersection = 0.0
        hXi = 0.0
        for j in range(y_test.shape[1]):
            hXi = hXi + int(predictions[i,j])
            if int(y_test[i,j]) == 1 and int(predictions[i,j]) == 1:
                intersection += 1
            
        if hXi != 0:
            precision = precision + float(intersection/hXi)
            

    precision = float(precision/y_test.shape[0])
    
    return precision


def recall(y_test, predictions):
    """
    Recall of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    recall : float
        recall of our model
    """
    recall = 0.0

    for i in range(y_test.shape[0]):
        intersection = 0.0
        Yi = 0.0
        for j in range(y_test.shape[1]):
            Yi = Yi + int(y_test[i,j])

            if y_test[i,j] == 1 and int(predictions[i,j]) == 1:
                intersection = intersection + 1
    
        if Yi != 0:
            recall = recall + float(intersection/Yi)    
    
    recall = recall/y_test.shape[0]
    return recall



def fbeta(y_test, predictions, beta=1):
    """
    FBeta of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    fbeta : float
        fbeta of our model
    """
    pr = precision(y_test, predictions)
    re = recall(y_test, predictions)

    num = float((1+pow(beta,2))*pr*re)
    den = float(pow(beta,2)*pr + re)

    if den != 0:
        fbeta = num/den
    else:
        fbeta = 0.0
    return fbeta
def relevantIndexes(matrix, row):
    """
    Gets the relevant indexes of a vector
    """
    relevant = []
    for j in range(matrix.shape[1]):
        if matrix[row,j] == 1:
            relevant.append(int(j))
    
    return relevant


def irrelevantIndexes(matrix, row):
    """
    Gets the irrelevant indexes of a vector
    """
    irrelevant = []
    for j in range(matrix.shape[1]):
        if matrix[row,j] == 0:
            irrelevant.append(int(j))
    
    return irrelevant

def multilabelConfussionMatrix(y_test, predictions):
    """
    Returns the TP, FP, TN, FN
    """
    TP = np.zeros(y_test.shape[1])
    FP = np.zeros(y_test.shape[1])
    TN = np.zeros(y_test.shape[1])
    FN = np.zeros(y_test.shape[1])

    for j in range(y_test.shape[1]):
        TPaux = 0
        FPaux = 0
        TNaux = 0
        FNaux = 0
        for i in range(y_test.shape[0]):
            if int(y_test[i,j]) == 1:
                if int(y_test[i,j]) == 1 and int(predictions[i,j]) == 1:
                    TPaux += 1
                else:
                    FPaux += 1
            else:
                if int(y_test[i,j]) == 0 and int(predictions[i,j]) == 0:
                    TNaux += 1
                else:
                    FNaux += 1
        TP[j] = TPaux
        FP[j] = FPaux
        TN[j] = TNaux
        FN[j] = FNaux

    return TP, FP, TN, FN

def multilabelMicroConfussionMatrix(TP, FP, TN, FN):
    TPMicro = 0.0
    FPMicro = 0.0
    TNMicro = 0.0
    FNMicro = 0.0
    
    for i in range(len(TP)):
        TPMicro = TPMicro + TP[i]
        FPMicro = FPMicro + FP[i]
        TNMicro = TNMicro + TN[i]
        FNMicro = FNMicro + FN[i]
    
    return TPMicro, FPMicro, TNMicro, FNMicro

def rankingMatrix(probabilities):
    """
    Matrix with the rankings for each label
    """
    ranking = np.zeros(shape=[probabilities.shape[0], probabilities.shape[1]])
    probCopy = np.copy(probabilities)
    for i in range(probabilities.shape[0]):
        indexMost = 0
        iteration = 1
        while(sum(probCopy[i,:]) != 0):
            for j in range(probabilities.shape[1]):
                if probCopy[i,j] > probCopy[i,indexMost]:
                    indexMost = j
                ranking[i, indexMost] = iteration
                probCopy[i, indexMost] = 0
                iteration += 1
    
    return ranking
def oneError(y_test, probabilities):
    """
    One Error 
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    probabilities: sparse or dense matrix (n_samples, n_labels)
        Probability of being into a class or not per each label
    Returns
    =======
    oneError : float
        One Error
    """
    oneerror = 0.0
    ranking = rankingMatrix(probabilities)

    for i in range(y_test.shape[0]):
        index = np.argmin(ranking[i,:])
        if y_test[i,index] == 0:
            oneerror += 1.0
    
    oneerror = float(oneerror)/float(y_test.shape[0])

    return oneerror

def coverage(y_test, probabilities):
    """
    Coverage
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    probabilities: sparse or dense matrix (n_samples, n_labels)
        Probability of being into a class or not per each label
    Returns
    =======
    coverage : float
        coverage
    """
    coverage = 0.0
    ranking = rankingMatrix(probabilities)

    for i in range(y_test.shape[0]):
        coverageMax = 0.0
        for j in range(y_test.shape[1]):
            if y_test[i,j] == 1:
                if ranking[i,j] > coverageMax:
                    coverageMax = ranking[i,j]
        
        coverage += coverageMax

    coverage = float(coverage)/float(y_test.shape[0])
    coverage -= 1.0

    return coverage

def averagePrecision(y_test, probabilities):
    """
    Average Precision
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    probabilities: sparse or dense matrix (n_samples, n_labels)
        Probability of being into a class or not per each label
    Returns
    =======
    averageprecision : float
        Average Precision
    """
    averageprecision = 0.0
    averageprecisionsummatory = 0.0
    ranking = rankingMatrix(probabilities)
    
    for i in range(y_test.shape[0]):
        relevantVector =relevantIndexes(y_test, i)
        for j in range(y_test.shape[1]):
            average = 0.0
            if y_test[i, j] == 1:
                for k in range(y_test.shape[1]):
                    if(y_test[i,k] == 1):
                        if ranking[i,k] <= ranking[i,j]:
                            average += 1.0
            if ranking[i,j] != 0:
                averageprecisionsummatory += average/ranking[i,j]
        
        if len(relevantVector) == 0:
            averageprecision += 1.0
        else:
            averageprecision += averageprecisionsummatory/float(len(relevantVector))
        averageprecisionsummatory = 0.0
    
    averageprecision /= y_test.shape[0]
    return averageprecision

def rankingLoss(y_test, probabilities):
    """
    Ranking Loss
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    probabilities: sparse or dense matrix (n_samples, n_labels)
        Probability of being into a class or not per each label
    Returns
    =======
    rankingloss : float
        Ranking Loss
    """
    rankingloss = 0.0

    for i in range(y_test.shape[0]):
        relevantVector = relevantIndexes(y_test, i)
        irrelevantVector = irrelevantIndexes(y_test, i)
        loss = 0.0

        for j in range(y_test.shape[1]):
            if y_test[i,j] == 1:
                for k in range(y_test.shape[1]):
                    if y_test[i,k] == 0:
                        if float(probabilities[i,j]) <= float(probabilities[i,k]):
                            loss += 1.0
        if len(relevantVector) != 0 and len(irrelevantVector) != 0:
            rankingloss += loss/float(len(relevantVector)*len(irrelevantVector))
    
    rankingloss /= y_test.shape[0]

    return rankingloss
def accuracyMacro(y_test, predictions):
    """
    Accuracy Macro of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    accuracymacro : float
        Accuracy Macro of our model
    """
    accuracymacro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    for i in range(len(TP)):
        accuracymacro = accuracymacro + ((TP[i] + TN[i])/(TP[i] + FP[i] + TN[i] + FN[i]))
    
    accuracymacro = float(accuracymacro/len(TP))

    return accuracymacro


def accuracyMicro(y_test, predictions):
    """
    Accuracy Micro of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    accuracymicro : float
        Accuracy Micro of our model
    """
    accuracymicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    if (TPMicro + FPMicro + TNMicro + FNMicro) != 0:
        accuracymicro = float((TPMicro+TNMicro)/(TPMicro + FPMicro + TNMicro + FNMicro))

    return accuracymicro


def precisionMacro(y_test, predictions):
    """
    Precision Macro of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    precisionmacro : float
        Precision macro of our model
    """
    precisionmacro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    for i in range(len(TP)):
        if TP[i] + FP[i] != 0:
            precisionmacro = precisionmacro + (TP[i]/(TP[i] + FP[i]))

    precisionmacro = float(precisionmacro/len(TP))
    return precisionmacro


def precisionMicro(y_test, predictions):
    """
    Precision Micro of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    precisionmicro : float
        Precision micro of our model
    """
    precisionmicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)
    if (TPMicro + FPMicro) != 0:
        precisionmicro = float(TPMicro/(TPMicro + FPMicro))


    return precisionmicro

def recallMacro(y_test, predictions):
    """
    Recall Macro of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    recallmacro : float
        Recall Macro of our model
    """
    recallmacro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    for i in range(len(TP)):
        if TP[i] + FN[i] != 0:
            recallmacro = recallmacro + (TP[i]/(TP[i] + FN[i]))

    recallmacro = recallmacro/len(TP)
    return recallmacro

def recallMicro(y_test, predictions):
    """
    Recall Micro of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    recallmicro : float
        Recall Micro of our model
    """
    recallmicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    if (TPMicro + FNMicro) != 0:
        recallmicro = float(TPMicro/(TPMicro + FNMicro))

    return recallmicro


def fbetaMacro(y_test, predictions, beta=1):
    """
    FBeta Macro of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    fbetamacro : float
        FBeta Macro of our model
    """
    fbetamacro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    
    for i in range(len(TP)):
        num = float((1+pow(beta,2))*TP[i])
        den = float((1+pow(beta,2))*TP[i] + pow(beta,2)*FN[i] + FP[i])
        if den != 0:
            fbetamacro = fbetamacro + num/den

    fbetamacro = fbetamacro/len(TP)
    return fbetamacro

def fbetaMicro(y_test, predictions, beta=1):
    """
    FBeta Micro of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    fbetamicro : float
        FBeta Micro of our model
    """
    fbetamicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    num = float((1+pow(beta,2))*TPMicro)
    den = float((1+pow(beta,2))*TPMicro + pow(beta,2)*FNMicro + FPMicro)
    fbetamicro = float(num/den)

    return fbetamicro

def show_metric(y_test, predicted,beta):
    print ("Subset Accuracy: " + str(subsetAccuracy(y_test, predicted)),"Hamming Loss: " + str(hammingLoss(y_test, predicted)),"Accuracy: " + str(accuracy(y_test, predicted)),"Precision: " + str(precision(y_test, predicted)),"Recall: " + str(recall(y_test, predicted)),"FBeta: " + str(fbeta(y_test, predicted, beta)))
    print ("One Error: " + str(oneError(y_test, predicted)),"Ranking Loss: " + str(rankingLoss(y_test, predicted)),"Coverage: " + str(coverage(y_test, predicted)),"Average Precision: " + str(averagePrecision(y_test, predicted)))
    print ("Accuracy Macro: " + str(accuracyMacro(y_test, predicted)),"Accuracy Micro: " + str(accuracyMicro(y_test, predicted)),"Precision Macro: " + str(precisionMacro(y_test, predicted)),"Precision Micro: " + str(precisionMicro(y_test, predicted)),"Recall Macro: " + str(recallMacro(y_test, predicted)),"Recall Micro: " + str(recallMicro(y_test, predicted)),"FBeta Macro: " + str(fbetaMacro(y_test, predicted, beta)),"FBeta Micro: " + str(fbetaMicro(y_test, predicted, beta)))
