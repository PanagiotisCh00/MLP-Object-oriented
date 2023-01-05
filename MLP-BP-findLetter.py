from typing import List
import random
from xmlrpc.client import Boolean, boolean
from math import e
import matplotlib.pyplot as plt
import pandas as pd
all_id=1
#MLP BP to recognize a letter from some of its attributes
# learns and tests from the data.txt
# this class represents a node in the neuron networks. Every neuron has a unique id 
class Node :
    id: int 
    
    def __init__(self):
        global all_id
        self.id=all_id
        all_id =all_id+1
    # Method that print the id of the neuron-node    
    def printNode(self):
        print("id: ",self.id,end = '')
        
 # this class represents a connection. It represents which node im connected to and its curren and previous weight       
class ConnectTo:
    node:Node
    weight:float
    prev_weight:float
    
    def __init__(self,*args):
        self.node=args[0]
        self.prev_weight=0
        
        if len(args)<2 : #if no arguments given then a random weight is initialized
            self.weight=random.uniform(-1.0,1.0)
        else:
            self.weight=args[1]
        

    # method that print the connection    
    def printConnection(self):
        self.node.printNode()
        print (" weight: ",self.weight)
        
# this class represents an input node,which is a node with input data (0/1) and a list of all the out going edges it has,dld all the nodes
# that is connected to with their weights(of class connectTo)
class InputNode(Node):
    input_data:int
    out_going : List[ConnectTo]
    
    def __init__(self,input:int):
        super().__init__()
        self.input_data=input
        self.out_going=list()
        
    # function pou add an edge for the input node
    def addAkmh(self,node:Node):
        connection= ConnectTo(node)
        self.out_going.append(connection)
        
    # function that gets a node that the input node is connected and changes the weight
    def changeWeight(self,node:Node,weight:float):
        for el in self.out_going :
            if el.node is node:
                el.weight=weight
    
    # function to print the node
    def print(self):
        print("id: ",self.id,"data: ",self.input_data,"")
        print("out going edges:")
        for el in self.out_going :
            el.printConnection()
        print("############################################")
        
# this class represents a hidden node,which is a node a list of all the out going and in going edges it has,dld all the nodes
# that is connected to and from with their weights(of class connectTo). Also, it has a mid_result which is its result and its delta
class HiddenNode(Node):
    in_going: List[ConnectTo]
    out_going: List[ConnectTo]
    mid_result:float
    delta:float 
    
    def __init__(self):
        super().__init__()
        self.in_going=list()
        self.out_going=list()
        self.mid_result=0
        self.delta=0.0
        
    # function that adds an edge. flag=false then its ingoing edge
    def addAkmh(self,node:Node, flag: Boolean):
        connection= ConnectTo(node)
        if(not flag):
            self.in_going.append(connection)
        else:
            self.out_going.append(connection)
            
    # function that gets a node that its connected to and changes its weight.
    def changeWeight(self,node:Node,weight:float, flag:Boolean):
        if not flag :
            for el in self.in_going :
                if el.node is node:
                    el.weight=weight
        else:
            for el in self.out_going:
                if el.node is node:
                    el.weight=weight

    # function to print the node
    def print(self):
        print("id: ",self.id,"mid result: ",self.mid_result," delta: ",self.delta)
        print("in going:")
        for el in self.in_going :
            el.printConnection()
          
        print("out going:")
        for el in self.out_going :
            el.printConnection()
        print("############################################")
# this class represents a bias node,which is a node with input data(which will be always 1) and a list of all the out going edges it has,dld all the nodes
# that is connected to with their weights(of class connectTo)
class BiasNode(Node):
    out_going: List[ConnectTo]
    input_data:int
    
    def __init__(self):
        super().__init__()
        self.out_going=list()
        self.input_data=1
    # function to add an edge     
    def addAkmh(self,node:Node):
        connection= ConnectTo(node)
        self.out_going.append(connection)
    # function to print the node
    def print(self):
        print("id: ",self.id,"input ",self.input_data)          
        print("out going:")
        for el in self.out_going :
            el.printConnection()
        print("############################################")

        
# this class represents an output node,which has a list of all the in going edges it has,dld all the nodes
# that is connected with with their weights(of class connectTo), the final result of the node and its delta
class OutputNode(Node):
    in_going: List[ConnectTo]
    final_result: float 
    delta:float
    
    def __init__(self):
        super().__init__()
        self.in_going=list()
        self.final_result=0
        self.delta=0.0
        
    # function that adds an edge in the node
    def addAkmh(self,node:Node):
        connection= ConnectTo(node)
        self.in_going.append(connection)
        
    # function that gets a node that is connected and changes its weight
    def changeWeight(self,node:Node,weight:float):
        for el in self.in_going :
            if el.node is node:
                el.weight=weight
                
    # function to print the node
    def print(self):
        print("id: ",self.id,"result: ",self.final_result," delta: ",self.delta)
        print("in going:")
        for el in self.in_going :
            el.printConnection()
        print("############################################")

# function that is the sigmoid Function
def calcSigMoid(x):
    return (1/(1+pow(e,(-x))))



# class for the actual network. It has an input,hidden_one,hidden_two,output and bias layer where each list has types of the correct objects.
# also,it has the size of the hidden 2 layer,the learning rate,momentum. It has the all_input and all_output which are the input output of the train data,
# and it has the test_all_input,test_all_output for the input,output of the testing data.
class Network:
    input_layer : List[InputNode]
    hidden_one_layer:List[HiddenNode]
    hidden_two_layer:List[HiddenNode]
    hidden_two_size:int
    output_layer:List[OutputNode]
    bias_layer: List[BiasNode]
    learning_rate: float
    momentum:float
    all_input: List[List]
    all_output: List
    test_all_input: List[List]
    test_all_output: List
    
    def __init__(self,h1_size:int,h2_size:int,input_size:int,output_size:int,learning_rate:float,momentum:float, input_data, output_data,test_input_data,test_output_data):
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.input_layer = list()
        self.hidden_one_layer=list()
        self.hidden_two_layer=list()
        self.hidden_two_size=h2_size
        self.output_layer=list()
        self.bias_layer=list()
        self.all_input=list(list())
        self.all_input=input_data
        self.all_output=list()
        self.all_output=output_data
        self.test_all_input=list(list())
        self.test_all_input=test_input_data
        self.test_all_output=list()
        self.test_all_output=test_output_data
         # create input layer
        for i in range(0,input_size) :
            self.input_layer.append(InputNode(input_data[0][i]))
        # create hidden 1 layer 
        for i in range(0,h1_size) :
            self.hidden_one_layer.append(HiddenNode())
        # create hidden 2 layer 
        for i in range(0,h2_size) :
            self.hidden_two_layer.append(HiddenNode())
        # create output layer 
        for i in range(0,output_size) :
            self.output_layer.append(OutputNode())
        
        # Add all the edges(i use <-> because i add all the edges both ways)
        # input to (<->) hidden 1 layer
        for n1 in self.input_layer:
            for n2 in self.hidden_one_layer :
                connection_to=ConnectTo(n2)
                connection_from= ConnectTo(n1,connection_to.weight)
                n1.out_going.append(connection_to)
                n2.in_going.append(connection_from)
        
        # from hidden 1 <-> hidden 2
        if h2_size > 0:
            for n1 in self.hidden_one_layer:
                for n2 in self.hidden_two_layer:
                    connection_to=ConnectTo(n2)
                    connection_from= ConnectTo(n1,connection_to.weight)
                    n1.out_going.append(connection_to)
                    n2.in_going.append(connection_from)
            # from hidden 2 <-> output
            for n1 in self.hidden_two_layer:
                for n2 in self.output_layer:
                    connection_to=ConnectTo(n2)
                    connection_from= ConnectTo(n1,connection_to.weight)
                    n1.out_going.append(connection_to)
                    n2.in_going.append(connection_from)
        # from hidden 1 <-> output
        for n1 in self.hidden_one_layer:
            for n2 in self.output_layer:
                connection_to=ConnectTo(n2)
                connection_from= ConnectTo(n1,connection_to.weight)
                n1.out_going.append(connection_to)
                n2.in_going.append(connection_from)
        
        # add the bias neurons and connect them with the nodes of the layers. the first bias connects with the hidden 1 layer's neurons,
        # the second with the hidden 2 or output layer's neurons,
        # and the third connects with the output layer's neurons
        self.bias_layer.append(BiasNode())
        for n1 in self.hidden_one_layer :
            connection_from=ConnectTo(self.bias_layer[0])
            n1.in_going.append(connection_from)
            connection_to=ConnectTo(n1,connection_from.weight)
            self.bias_layer[0].out_going.append(connection_to)
        if h2_size>0 :
            self.bias_layer.append(BiasNode())
            for n1 in self.hidden_two_layer :
                connection_from=ConnectTo(self.bias_layer[1])
                n1.in_going.append(connection_from)
                connection_to=ConnectTo(n1,connection_from.weight)
                self.bias_layer[1].out_going.append(connection_to)
           
            self.bias_layer.append(BiasNode())
            for n1 in self.output_layer :
                connection_from=ConnectTo(self.bias_layer[2])
                n1.in_going.append(connection_from)
                connection_to=ConnectTo(n1,connection_from.weight)
                self.bias_layer[2].out_going.append(connection_to)
        else :
            self.bias_layer.append(BiasNode())
            for n1 in self.output_layer :
                connection_from=ConnectTo(self.bias_layer[1])
                n1.in_going.append(connection_from)
                connection_to=ConnectTo(n1,connection_from.weight)
                self.bias_layer[1].out_going.append(connection_to)

                            
    # method to print the whole network ,mostly for my usage to check that the network was created correctly        
    def printNetwork(self):
        print("Network\n")
        print("Input Layer")
        for el in self.input_layer :
           el.print() 
        print("Hidden 1 Layer")
        for el in self.hidden_one_layer :
           el.print() 
        if  self.hidden_two_size>0:
            print("hidden 2 Layer")
            for el in self.hidden_two_layer :
                el.print() 
        print("Output Layer")
        for el in self.output_layer :
           el.print() 
        print("Bias Layer")
        for el in self.bias_layer :
           el.print() 
    
        
    # function for the forward  pass to find the results of hidden and output(real output) and add them to the nodes
    # flag indicates if its for testing or training to get the correct input data lists
    def doForwardPhase(self,iter,flag:boolean):
        # add the input data to the neurons of input layer:
        if not flag:
            position=0
            for el in self.input_layer:
                el.input_data=(self.all_input[iter][position])
                position=position+1 
        else:
            position=0
            for el in self.input_layer:
                el.input_data=(self.test_all_input[iter][position])
                position=position+1 
            
        # find the result of all the nodes of hidden layer 1 :
        for el in self.hidden_one_layer :
            sum =0
            for edge in el.in_going :
                sum =sum + (edge.weight * edge.node.input_data)
            el.mid_result=calcSigMoid(sum)
        # find the result of all the nodes of hidden layer 2 :
        if  self.hidden_two_size>0:
            for el in self.hidden_two_layer :
                sum =0
                for edge in el.in_going :
                    if (isinstance(edge.node,BiasNode)) : # this if statement exist because the bias nodes have other name for their input
                        sum =sum + (edge.weight * edge.node.input_data)
                    else:    
                        sum =sum + (edge.weight * edge.node.mid_result)
                el.mid_result=calcSigMoid(sum)

        # find the result of  the nodes of the output layer:
        # c_temp=1
        for el in self.output_layer :
            sum =0   
           # print("alo node, ",c_temp)
            for edge in el.in_going :
                if (isinstance(edge.node,BiasNode)) : # this if statement exist because the bias nodes have other name for their input
                    sum =sum + (edge.weight * edge.node.input_data)
                   # print("bias:",sum)
                else:    
                    sum =sum + (edge.weight * edge.node.mid_result)
                   # print("not bias:",sum)                  
           #  c_temp=c_temp+1
            el.final_result=calcSigMoid(sum) 
            # print(el.final_result )
        # find the accuracy,if its correct or wrong the predicted
        max_predicted_result=0.0
        neuron_num=-1
        counter=0
        # print(self.output_layer[0].final_result)
        for el in self.output_layer:
            if(max_predicted_result<el.final_result):
                max_predicted_result=el.final_result
                neuron_num=counter
            counter=counter+1
                    
        if flag :
            # if the output in the iteration im at and if the neuron i found that is activated by my function is 1 thena accuracy is 1
            if (self.test_all_output[iter][neuron_num]==1):
                accuracy=1
            else:
                accuracy=0
            s=list()
            # find the error,which is the mean least square as we learned in class.
            err=0
            counter=0
            for el in self.test_all_output[iter]:
                err=err+((el- self.output_layer[counter].final_result)* ((el- self.output_layer[counter].final_result)))
                counter=counter+1
                    
            s.append(err)
            s.append(accuracy)
            return s
        else:
            # if the output in the iteration im at and if the neuron i found that is activated by my function is 1 thena accuracy is 1 for testing
            if (self.all_output[iter][neuron_num]==1):
                accuracy=1
            else:
                accuracy=0
            err=0
            counter=0    
            for el in self.all_output[iter]:
                err=err+((el- self.output_layer[counter].final_result)* ((el- self.output_layer[counter].final_result)))
                counter=counter+1
            s=list()
            s.append(err)
            s.append(accuracy)
            
            return s
        # self.printNetwork()
    
    # this method does the back propagation stage of the algorithm    
    def doBackwardPhase(self,iter):
        target_result=self.all_output[iter]
        # start from output
        # calculate the delta of the output layer
        counter=0
        for el in self.output_layer:
            temp=el.final_result*(1-el.final_result)*(el.final_result-target_result[counter])
            el.delta=(temp)
            counter=counter+1
        # calculate the delta of each node of the second hidden layer if it exists    
        if(self.hidden_two_size>0):
            for el in self.hidden_two_layer:
                # find sum of deltas with weights
                sum=0
                for next in el.out_going:
                    sum=sum+ (next.node.delta*next.weight)
                el.delta=(el.mid_result*(1-el.mid_result)*sum)
         # calculate the delta of each node of the first hidden layer
        for el in self.hidden_one_layer:
            # find sum of deltas with weights
            sum=0
            for next in el.out_going:
                sum=sum+(next.node.delta*next.weight)
            el.delta=(el.mid_result*(1-el.mid_result)*sum)
            
        
    # this method does the adaptation of weights stage of the algorithm    
    def adaptWeights(self,iter):
        # start by the input nodes
        # i find the new weight of every node in input layer change it ,change the previous weight, and for every weight of each edge
        # i change the in going weight of the node that is connected so that all the nodes have the correct and same information
        for el in self.input_layer:
            for edge in el.out_going :
                temp=edge.weight-(self.learning_rate*edge.node.delta*el.input_data)+(self.momentum*(edge.weight-edge.prev_weight))
                edge.prev_weight=edge.weight
                edge.weight=temp 
                next_node=edge.node
                for to_edges in next_node.in_going:
                    if(to_edges.node is el):
                        to_edges.prev_weight=to_edges.weight
                        to_edges.weight=temp

        # change the weights of the first hidden layer neurons with the same logic
        for el in self.hidden_one_layer:
            for edge in el.out_going:
                temp=edge.weight-(self.learning_rate*edge.node.delta*el.mid_result)+(self.momentum*(edge.weight-edge.prev_weight))
                edge.prev_weight=edge.weight
                edge.weight=temp
                next_node=edge.node
                for to_edges in next_node.in_going:
                    if(to_edges.node is el):
                        to_edges.prev_weight=to_edges.weight
                        to_edges.weight=temp
                
        # change the weights of the hidden layer 2 neurons with the same logic
        if self.hidden_two_size>0 :
            for el in self.hidden_one_layer:
                for edge in el.out_going:
                    temp=edge.weight-(self.learning_rate*edge.node.delta*el.mid_result)+(self.momentum*(edge.weight-edge.prev_weight))
                    edge.prev_weight=edge.weight
                    edge.weight=temp 
                    next_node=edge.node
                    for to_edges in next_node.in_going:
                        if(to_edges.node is el):
                            to_edges.prev_weight=to_edges.weight
                            to_edges.weight=temp    
        # bias layer        
        # change the weights of the nodes of the bias layer.
        # bias[0] with hidden 1 
        for edge in self.bias_layer[0].out_going:
            temp=edge.weight-(self.learning_rate*edge.node.delta)+(self.momentum*(edge.weight-edge.prev_weight))
            edge.prev_weight=edge.weight
            edge.weight=temp
            next_node=edge.node
            for to_edges in next_node.in_going:
                if(to_edges.node is self.bias_layer[0]):
                    to_edges.prev_weight=to_edges.weight
                    to_edges.weight=temp
        if self.hidden_two_size>0:
            # bias[1]-->hidden2
            for edge in self.bias_layer[1].out_going:
                temp=edge.weight-(self.learning_rate*edge.node.delta)+(self.momentum*(edge.weight-edge.prev_weight))
                edge.prev_weight=edge.weight
                edge.weight=temp
                next_node=edge.node
                for to_edges in next_node.in_going:
                    if(to_edges.node is self.bias_layer[1]):
                        to_edges.prev_weight=to_edges.weight
                        to_edges.weight=temp
             # bias[2]-->output
            for edge in self.bias_layer[2].out_going:
                temp=edge.weight-(self.learning_rate*edge.node.delta)+(self.momentum*(edge.weight-edge.prev_weight))
                edge.prev_weight=edge.weight
                edge.weight=temp
                next_node=edge.node
                for to_edges in next_node.in_going:
                    if(to_edges.node is self.bias_layer[2]):
                        to_edges.prev_weight=to_edges.weight
                        to_edges.weight=temp
        else:
            # bias[1]->output
            for edge in self.bias_layer[1].out_going:
                temp=edge.weight-(self.learning_rate*edge.node.delta)+(self.momentum*(edge.weight-edge.prev_weight))
                edge.prev_weight=edge.weight
                edge.weight=temp
                next_node=edge.node
                for to_edges in next_node.in_going:
                    if(to_edges.node is self.bias_layer[1]):
                        to_edges.prev_weight=to_edges.weight
                        to_edges.weight=temp
            
    
    # function that gets the iterations which is the number of epochs that we train and test our model.
    # then for all those epochs it trains the model using the forward,backward and adaptation of weights stages. After its trained it is being tested
    # with the forward phase and print the data the homework wants in errors.txt and succerate.txt        
    def trainAndTest(self,iterations):
        train_error=0
        # epoxes-iterations
        f = open("errors.txt", "w")
        f2=open("successrate.txt","w")
        import time
        start=time.time()
        testing_error_list=list();training_error_list=list()
        testing_accuracy_list=list();training_accuracy_list=list()
        epochs_list=list()
        for i in range (0,iterations):
            # st=time.time()
            epochs_list.append(i)
           # first we train the model:
            train_error=0
            train_accuracy_score=0
            for j in range(0, len(self.all_input)):
                t=self.doForwardPhase(j,False)       
                train_error=train_error+t[0]
                train_accuracy_score=train_accuracy_score+t[1]
                self.doBackwardPhase(j)
                self.adaptWeights(j)
            testing_error=0.0
            test_accuracy_score=0
            for j in range(0, len(self.test_all_input)):
                t=self.doForwardPhase(j,True)
                testing_error=testing_error+t[0]
                test_accuracy_score=test_accuracy_score+t[1]
            # print("iteration ",(i+1),"train_error ",train_error/(4), "testing error: ",testing_error/4)
            train_size=len(self.all_input)
            test_size=len(self.test_all_input)
            s =str(i+1)+" "+str(train_error/train_size)+" "+str(testing_error/test_size)+"\n"
            testing_error_list.append(testing_error/test_size);training_error_list.append(train_error/train_size)
            f.write(s)
            # print("iteration ",(i+1),"train_accuracy ",100*train_accuracy_score/train_size, "testing accuracy: ",100*test_accuracy_score/test_size)
            testing_accuracy_list.append((100*test_accuracy_score/test_size));training_accuracy_list.append((100*train_accuracy_score/train_size))
            
            s2 =str(i+1)+" "+str(100*train_accuracy_score/train_size)+" "+str(100*test_accuracy_score/test_size)+"\n"
            f2.write(s2)
            
            # d=time.time()
            # print("epoch ",i)
            # print(st-d)

        f.close()
        f2.close()
        end=time.time()
        print("finished: ",end-start)
        plt.plot(epochs_list,training_error_list,label ="training error",color='red')
        plt.plot(epochs_list,testing_error_list,label='testing error',color='blue')
        plt.xlabel("Epochs")
        plt.ylabel("Men square error(mse)")
        plt.legend()
        plt.savefig('error.jpg')
        plt.close()
        
        plt.plot(epochs_list,training_accuracy_list,label ="training accuracy",color='red')
        plt.plot(epochs_list,testing_accuracy_list,label='testing accuracy',color='blue')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('accuracy.jpg')

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

# get a letter and returns a list with 26 elements and the correct element is 1 based on ALphabet:if A->1000..., B->01000...
def makeLetterToList(letter:str):
    res_list=list()
    for i in range(0,26):
        res_list.append(0)
    
    res_list[(ord(letter)-ord('A'))]=res_list[(ord(letter)-ord('A'))]+1
    return res_list
    
 
# method that is called when program starts
def startMethod () :
    # always the data are in data.txt as the exercise says.
    # it reads data
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
                        
    h1_size=-1;h2_size=-1;input_size=-1;output_size=-1;learning_rate=-1;momentum=-1;iterations=-1;
    file = open("parameters.txt")
    lines = file.readlines()
    for line in lines:
        data=line.split()
        parameter_name=data[0]
        value=data[1]    
        if parameter_name == 'numHiddenLayerOneNeurons':
            h1_size=int(value)
        elif (parameter_name == "numHiddenLayerTwoNeurons"):
            h2_size=int(value)
        elif (parameter_name == "numInputNeurons"):
            input_size=int(value)
        elif parameter_name == "numOutputNeurons":
            output_size=int(value)
        elif parameter_name == "learningRate":
            learning_rate=float(value)
        elif parameter_name == "momentum":
            momentum=float(value)
        elif parameter_name == "maxIterations":
            iterations=int(value)
        else:
            print("error, incorrect data given\n")
            return 
    if h1_size==-1 or h2_size==-1 or input_size==-1 or output_size==-1 or learning_rate==-1 or momentum==-1 or iterations==-1 :
            print("Not enough data given\n")
            return 
    print(h1_size," ",h2_size," ",input_size," ",output_size," ",learning_rate," ",momentum," ",iterations)
    # print(letter_list)
    # print(features_list)
    normalizeInputData(features_list)  
    # print(features_list)
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
            output_list.append(makeLetterToList(letter=letter))
            num_of_letters[(ord(letter)-ord('A'))]=num_of_letters[(ord(letter)-ord('A'))]-1
        else:
            test_input_list.append(features_list[counter])
            test_output_list.append(makeLetterToList(letter=letter))
        counter =counter+1
    # print(input_list);print(output_list);
    # print(test_input_list);print(test_output_list);     
            
    
    x=Network(h1_size=h1_size,h2_size=h2_size,input_size=input_size,output_size=output_size,learning_rate= learning_rate,momentum=
              momentum,input_data=input_list,output_data=output_list,test_input_data=test_input_list,test_output_data=test_output_list)
    x.trainAndTest(iterations=iterations)
    print("finished")



    
    

  
  
if __name__=="__main__":
    startMethod()
    