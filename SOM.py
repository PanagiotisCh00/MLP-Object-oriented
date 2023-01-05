from typing import List
import random
from xmlrpc.client import Boolean, boolean
from math import e
import math
import matplotlib.pyplot as plt
import pandas as pd
#Kohonnen SOM to recognize handwritten letters from data.txt

all_id=1
# this class represents a neuron of the network (output neuron). It has an id, a list of the weights and a list of the initial weights.
# also it has the label--letter that the neuron represent
class Node:
    id:int
    weights:list
    weight_initial:list 
    # i have the label that the neuron represent the letter(in beginning has no value)
    label="none"
    
    def __init__(self,num_features:int) :
        
        global all_id
        self.id=all_id
        all_id=all_id+1
        self.weights=list()
        self.weight_initial=list()
        # initialise the weights random of the neuron
        # weights : between 0-1
        for i in range(0,num_features):
            x=random.uniform(0,1.0)
            self.weight_initial.append(x)
            self.weights.append(x)
    def printNode(self):
        print("id ",self.id,"weight: ",self.weights)
        
# represents the network that has a lilst of nodes, the training,testing list and a list about the results of the tasting that will be used to make the labeling.
class Network:
    # need of a 2-D list of nodes
    nodes:list
    training_list:list
    testing_list:list
    test_list_result:list 
    dimension:int
    def __init__(self,dimension,num_features,training,testing,test_result) :
        self.nodes=list()
        self.training_list=training
        self.test_list_result=test_result
        self.testing_list=testing
        self.dimension=dimension
        for i in range(0,dimension):
            row_list=list()
            for j in range(0,dimension):
                row_list.append(Node(num_features))
            self.nodes.append(row_list)
    def printNetwork(self):
        for row in self.nodes:
            for col in row:
                col.printNode()
    
    # for every epoch it trains the network and then it tests it
    # it uses the types,formulas as discussed in the course. First finds the winner,
    # then based on it calculate every new weight
    def trainTest(self,epochs:int,learning_rate:float):
        f=open("result.txt",'w');
        training_error_list=list();testing_error_list=list();
        s_initial=len(self.nodes)/2; learning_rate_initial=learning_rate
        s=self.dimension/2 #arxiko σ
        epoch_num=1
        # for all the epochs
        for k in range(0,epochs):
            sum_d_training=0
            # for each data in training
            for data in self.training_list:
                min_d=float('inf')
                x_min=-1;y_min=-1
                # to find minimum distance--neuron from input
                for i in range(0,self.dimension) :
                    for j in range(0,self.dimension):
                        d=0
                        counter=0
                        # for all the attributes-weights for the node
                        for el in self.nodes[i][j].weights:
                            d=d+(data[counter]-el)*(data[counter]-el)
                            counter=counter+1
                        if(d<min_d): #find smallest distance of weights
                            min_d=d
                            x_min=i;y_min=j;
                sum_d_training=sum_d_training+min_d
                # now i find hci and change weights for each neuron
                for i in range(0,self.dimension) :
                    for j in range(0,self.dimension):
                        hci= pow(e,-(((x_min-i)*(x_min-i))+((y_min-j)*(y_min-j))/(2*(s*s))))
                        for w in range(0,len(self.nodes[i][j].weights)):
                            # self.nodes[i][j].weights[w]=self.nodes[i][j].weights[w]+(learning_rate * hci * (data[w]-self.nodes[i][j].weight_initial[w]))
                            self.nodes[i][j].weights[w]=self.nodes[i][j].weights[w]+(learning_rate * hci * (data[w]-self.nodes[i][j].weights[w]))
                
            sum_d_testing=0
            # for each data in testing
            for data in self.testing_list:
                min_d=float('inf')
                x_min=-1;y_min=-1
                # to find minimum distance--neuron from input
                for i in range(0,self.dimension) :
                    for j in range(0,self.dimension):
                        d=0
                        counter=0
                        for el in self.nodes[i][j].weights:
                            d=d+(data[counter]-el)*(data[counter]-el)
                            counter=counter+1
                        if(d<min_d):
                            min_d=d
                            x_min=i;y_min=j;
                sum_d_testing=sum_d_testing+min_d
            # adaptation of learning rate and s(σ) after every epoch
            learning_rate=learning_rate_initial * pow(e,-(epoch_num/epochs))
            s=s_initial * pow(e, -(epoch_num/(epochs/math.log(s_initial))))    
                        
            training_error=(sum_d_training*sum_d_training)/len(self.training_list)
            training_error_list.append(training_error)
            testing_error=(sum_d_testing*sum_d_testing)/len(self.testing_list)
            testing_error_list.append(testing_error)
            wrt_data =str(epoch_num)+" "+str(training_error)+" "+str(testing_error)+ "\n"
            f.write(wrt_data)
            print(wrt_data)
            epoch_num=epoch_num+1
        # after it finished all the epochs we do the labeling
        
        # for every output node i found the letter-input has the smallest distance to
        for i in range(0,self.dimension) :
            for j in range(0,self.dimension):
                min_d=float('inf')
                counter=0
                c=0
                for data in self.testing_list:
                    d=0;counter=0;
                    for el in self.nodes[i][j].weights:
                        d=d+(pow(data[counter]-el,2))
                        counter=counter+1
                    if(d<min_d):
                        min_d=d
                        self.nodes[i][j].label=self.test_list_result[c]
                    c=c+1
        f2=open("clustering.txt",'w');
        for i in range(0,self.dimension) :
            for j in range(0,self.dimension):
                str_2="("+str(i)+","+str(j)+") "+self.nodes[i][j].label+ "\n"
                f2.write(str_2)
            
                                
                    
        

    
    

# gets a 2-D list and goes vertical for each feature to do normalization        
def normalizeInputData(features_list:list):
    # run the list vertically to find min,max
    # find max in list
    max_list=list()
    min_list=list()
    for i in range (0,16):
        max_list.append(pd.DataFrame(features_list)[i].max())
        min_list.append(pd.DataFrame(features_list)[i].min())

    pos=0
    for i in range(len(features_list[0])):
        for x in features_list:
            x[i]=(x[i]-min_list[pos])/(max_list[pos]-min_list[pos])
        pos=pos+1

    

def startMethod():
    file = open("data.txt")
    lines = file.readlines()
    letter_list=list()
    features_list=list(list())
    for line in lines:
        data=line.split(",")
        letter_list.append(data[0])
        temp_list=list()
        for i in range (1,16):
            temp_list.append(int(data[i]))
        temp_list.append(int(data[16].split("\n")[0]))
        features_list.append(temp_list)
    normalizeInputData(features_list)  
    # find number of occurences of each letter add to list (goes alphabetically:a->0)
    num_of_letters=list()
    for i in range (0,26):
        num_of_letters.append(0)
    for el in letter_list:
        num_of_letters[(ord(el)-ord('A'))]=num_of_letters[(ord(el)-ord('A'))]+1
    # print(num_of_letters)
    # now i will replace each number of occurences with its 2/3 
    for i in range (len(num_of_letters)):
        num_of_letters[i]=num_of_letters[i]*2/3
    input_list=list();output_list=list();
    test_input_list=list();test_output_list=list();
    counter=0
    for letter in letter_list :
        # im before 2/3 s its training
        if(num_of_letters[(ord(letter)-ord('A'))]>0):
            input_list.append(features_list[counter])
            output_list.append(letter)
            num_of_letters[(ord(letter)-ord('A'))]=num_of_letters[(ord(letter)-ord('A'))]-1
            
        else:
            test_input_list.append(features_list[counter])
            test_output_list.append(letter)
        counter =counter+1
    # print(input_list);print(output_list);
    # print(test_input_list);print(test_output_list);  
   
    epochs=50
    learning_rate=0.7
    dimension=50
    epochs = int(input("Give number of iterations (epochs): "))
    learning_rate=float(input("Give the learning rate"))
    print(epochs, " ",learning_rate)
    # 16 features standard from exercise-data.txt
    num_feats=16 
    #print(input_list)
    #print("tee\n",test_input_list)
    network=Network(dimension=dimension,num_features=num_feats,training=input_list,testing=test_input_list,test_result=test_output_list)
    # network.printNetwork()
    network.trainTest(epochs=epochs, learning_rate=learning_rate)


  
  
if __name__=="__main__":
    startMethod()