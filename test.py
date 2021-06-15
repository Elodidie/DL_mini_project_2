import torch
import math
from torch import empty
import time
import datetime
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

import framework


#--------------------------------------------------#   
### Util functions ###

def generate_data(): 
    """
    Generates a data set
    Ouputs:
        train_input : 1000 points sampled uniformly in [0, 1]^2
        train_label : labels whether 
    """
    train_input = torch.empty(1000,2).uniform_(0,1)
    radius=1/(2*math.pi)
    center=torch.Tensor([0.5, 0.5])
    train_label=((train_input-center).pow(2).sum(axis=1)<radius).long()
    return train_input, train_label

def evaluate_model(y_pred, y_true):
    """
    Assesses model performance
    Inputs:
        y_pred : model's prediction
        y_true : true labels
    Ouputs:
        accuracy, F1 score, recall, specificity, precision
    """
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(y_true.size()[0]):
        if y_true[i].item()==1:
            if y_pred[i]==y_true[i].item():
                TP+=1
            else:
                FN+=1
        if y_true[i].item()==0:
            if y_pred[i]==y_true[i].item():
                TN+=1
            else:
                FP+=1
    print('TP: ', TP, ' TN: ', TN, ' FP: ', FP, ' FN: ', FN)            
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    F1_score=2*TP/(2*TP+FP+FN)
    recall=TP/(TP+FN)
    specificity=TN/(TN+FP)
    precision=TP/(TP+FP)
        
    return accuracy, F1_score, recall, specificity, precision

def predict(model, test_input):
    """
    Computes prediction from the model, over a data set
    Inputs:
        model : trained model
        test_input : data set to compute the prediction labels from
    Ouputs:
        prediction labels as a list 
    """
    predictions=[]
    for n in range(test_input.size(0)):
            
        prediction = model.forward(test_input[n])
        
        #binarizer
        if prediction>0:
            prediction=torch.Tensor([1])
        else:
            prediction=torch.Tensor([0])
        
        predictions.append(int(prediction))
        
    return predictions

#--------------------------------------------------#   
### Training ###

def train_model(train_data, train_labels, test_data, test_labels,
                    model, learning_rate, nb_epochs, loss_f = 'MSE'):
    """
    Trains a given model
    Ouputs:
        model : the model after training
        train_errors : list of training errors computed at each epoch
    """

    # Initialize optimizer
    SGD_ = framework.SGD(model, learning_rate)
    
    # Initialize empty lists
    train_errors = []
    start_time = time.time()
    print('## TRAINING STARTED##')
    
    # Initialize loss
    if loss_f == 'MSE':
        Loss = framework.LossMSE()
    elif loss_f == 'MAE':
        Loss = framework.LossMAE()
    else:
        raise "Enter a valid loss function (accepted names are 'MSE' or 'MAE')"
    
    
    for e in range(nb_epochs):
        
        # Reset training values at each epoch
        training_loss = 0
        nb_train_errors = 0
        

        for n in range(train_data.size(0)):
            
            # Clear gradients
            SGD_.zero_grad()
            
            # Output layer
            prediction = model.forward(train_data[n])
            #print('prediction before and label',prediction, train_data[n])
            
            # Binary step
            if prediction>0:
                prediction=torch.Tensor([1])
            else:
                prediction=torch.Tensor([0])
            
            # Add error if prediction is wrong
            if int(train_labels[n].item()) != prediction: 
                nb_train_errors += 1

            # Compute loss
            training_loss +=Loss.forward(prediction, train_labels[n].float())
            d_loss = Loss.backward()

            # Backpropogation step
            model.backward(d_loss)

            # Update model with SGD step
            SGD_.step()
            
         # Store training accuracy for this epoch
        train_acc = (100 * nb_train_errors) / train_data.size(0) 
        train_errors.append(train_acc)        
        

        training_accuracy = 100-((100 * nb_train_errors) / train_data.size(0))
        print(e+1, ' training accuracy: ',training_accuracy)
        

    # Training time
    end_time = time.time()
    training_time = int(end_time - start_time)
    print("Training time : {:3}\n"
          .format(str(datetime.timedelta(seconds = training_time))))
    
    return model, train_errors


#--------------------------------------------------#   
### Testing ###

def test(model, nb_trials, learning_rate, nb_epochs, loss_function):
    """
    Run the process of training and evaluating performance over a number nb_trials of trials, 
        for statistics purposes
    Ouputs:
        test_input : test input of the last trial
        prediction : prediction over test_input
        train_error_evolution : training output of the last trial
        ... : diverse metrics sastistics over the n trials
    """
    accuracies = []
    F1_scores = []
    train_error_evolution = []
    train_input, train_label, test_input, test_label = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
    predictions = torch.Tensor([0])
    
    for n in range(nb_trials):
        #generate different shuffles
        train_input, train_label = generate_data()
        test_input, test_label = generate_data()
        
        # train model for each one. Returns model after training, and list of training & testing error evolution
        model, train_errors = train_model(train_input, train_label, test_input, test_label, 
                         model, learning_rate, nb_epochs, loss_function) #est-ce qu'on mettrait pas un mode spécial print pour récupérer le plot de l'évolution de l'accuracy en fonction des epochs? en mm temps relou pcq calculer l'accuracy à chaque step, ça fait run le model sur les sets à chaque step
        predictions=predict(model, test_input)
        # store evolution
        train_error_evolution.append(train_errors)
        # compute metrics
        accuracy, F1_score, _, _, _ = evaluate_model(predictions, test_label)
        accuracies.append(accuracy)
        F1_scores.append(F1_score)
    
    # prediction over 
    prediction = predict(model, test_input)
    
    # metrics statistics
    mean_acc = (torch.tensor(accuracies)).mean().item()
    std_acc = (torch.tensor(accuracies)).std().item()
    mean_F1 = (torch.tensor(F1_scores)).mean().item()
    std_F1 = (torch.tensor(F1_scores)).std().item()
    
    return test_input, prediction, train_error_evolution, accuracies, mean_acc, std_acc, F1_scores, mean_F1, std_F1


# Create model
model = framework.Model()

# Define learning rate and number of epochs
lr=0.001
e=25

# Build a network with two input units, one output unit, three hidden layers of 25 units
model.add(framework.Linear(2, 25))
model.add(framework.ReLU())
model.add(framework.Linear(25, 25))
model.add(framework.tanh())
model.add(framework.Linear(25, 25))
model.add(framework.ReLU())
model.add(framework.Linear(25, 1))


nb_trials = 1
test_input, prediction, train_error_evolution, accuracies, mean_acc, std_acc, F1_scores, mean_F1, std_F1 = test(model, nb_trials, lr,e, loss_function='MSE')