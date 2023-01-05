from typing import List
import random
from xmlrpc.client import Boolean, boolean
from math import e
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle 
neurons_id=1
random.seed(4)
# RBF networks that learns to recognize molecules

# a class for the centeres
# has the id, coordinates of point,coefficient,previous coefficient, sigma nad previous sigma
class Centers:
    id:int
    points:List # the actual centers "location"
    coeff:float
    prev_coeff:float
    sigma:float
    prev_sigma:float
    
    def __init__(self,sigma_input:float,point_input:List):
        global neurons_id
        self.id=neurons_id
        neurons_id=neurons_id+1
        self.coeff=random.uniform(-1.0,1.0)
        self.prev_coeff=0
        self.prev_sigma=0
        self.points=list()
        self.points=point_input
        self.sigma=sigma_input
    
    # function that prints the info of the center
    def printCenter(self):
        print("id: ",self.id," Points-centers: ",self.points, " Coeff ",self.coeff," Sigma ",self.sigma)
        
    # function that calculates the f and returns it
    def feta_function(self,pattern:List):
        num=0;
        for i in range(0,len(pattern)):
            num=num+(pow(pattern[i]-self.points[i],2))

        num=num/(2* pow(self.sigma,2))
        return (pow(e,-num))
 
# As there is always one output and the inputs neurons just send the input data, there is no need for objects for input or output or bias neurons
# Network class that has the training,testing lists, their results, a list of centers, the 3 learning rates , the bias coefficient and the number of epochs
class Network:
    training_list:List[List]
    training_results:List[float]
    testing_list:List[List]
    testing_results:List[float]
    centers_list:List[Centers]
    bias_coeff:float
    learning_r:float
    learning_sigma:float
    learning_weights:float
    iterations:int
    def __init__(self,training:List,training_res:List,testing:List,testing_res:List,center:List,
                    learning_rates:List,iters:int,num_of_hidden:int,all_sigmas:List):
        self.training_list=training;self.training_results=training_res;
        self.testing_list=testing;self.testing_results=testing_res;
        self.bias_coeff=random.uniform(-1.0,1.0)
        self.learning_r=learning_rates[0];self.learning_sigma=learning_rates[1];self.learning_weights=learning_rates[2];
        self.iterations=iters;
        self.centers_list=list()
        for i in range(0,num_of_hidden):
            self.centers_list.append(Centers(all_sigmas[i],center[i]))
     
    # function that prints info about the network
    def printNetwork(self):
        print("bias ",self.bias_coeff," Learning rate ",self.learning_r," learning sigma ",self.learning_sigma," learning weight ",self.learning_weights)
        for neuron in self.centers_list:
            neuron.printCenter()    
            print()
    # funciton that train,and tests the network for the number of epochs and creates the graphs
    def train_and_test(self):
        file  = open("results.txt",'w')
        train_error_list=list();
        test_error_list=list();
        epochs_list=list()
        for k in range(0, self.iterations):
            epochs_list.append(k)
            total_error=0
            # training:
            for i in range(0,len(self.training_list)):
                temp=0;
                # find real result
                feta_value_list=list()
                for neuron in self.centers_list:
                    feta_value=neuron.feta_function(self.training_list[i])
                    feta_value_list.append(feta_value)
                    temp=temp+(neuron.coeff * feta_value)
                real_result=self.bias_coeff+temp
                # print("res ",self.training_results[i], " real ",real_result)
                error=(self.training_results[i] - real_result) 
                # if i ==1 : print("error ",error);
                total_error=total_error+(pow(error,2))
               
                # update all Coefficients       
                # update weight of each neuron-center
                z=0
                for neuron in self.centers_list:
                    neuron.prev_coeff=neuron.coeff
                    neuron.coeff=neuron.coeff + (self.learning_weights * error * feta_value_list[z]) 
                    z=z+1
                          
                # update weight of bias
                self.bias_coeff=self.bias_coeff + (self.learning_weights * error)
                
                # update sigma of each center
                z=0
                for neuron in self.centers_list:
                    neuron.prev_sigma=neuron.sigma
                    temp_sub=0;
                    for j in range(0,len(neuron.points)):
                        temp_sub=temp_sub+pow(self.training_list[i][j]-neuron.points[j],2)
                    temp_sub=temp_sub/pow(neuron.sigma,3)
                    neuron.sigma=neuron.sigma +( self.learning_sigma * error * neuron.prev_coeff * feta_value_list[z] * temp_sub )
                    # print("before ",neuron.prev_sigma, " after ",neuron.sigma)
                    z=z+1
                
                # update coordinates of centers
                z=0
                for neuron in self.centers_list:
                    for j in range(0,len(neuron.points)):
                        val_temp=(self.training_list[i][j]-neuron.points[j]) / (pow(neuron.prev_sigma,2))
                        neuron.points[j]=neuron.points[j] + (self.learning_r * error * neuron.prev_coeff * feta_value_list[z] * val_temp)
                    z=z+1
            train_error_list.append(total_error/2)
            
            # testing 
            total_error_test=0
            for i in range(0,len(self.testing_list)):    
                temp=0;
                # find real result
                for neuron in self.centers_list:
                    temp=temp+(neuron.coeff * neuron.feta_function(self.testing_list[i]))
                real_result=self.bias_coeff+temp
                error=self.testing_results[i] - real_result
                total_error_test=total_error+(pow(error,2))
            test_error_list.append(total_error_test/2)
            
            str_var=str((k+1))+" "+str(total_error/2)+" "+ str(total_error_test/2)+"\n"
            file.write(str_var)
        
        # graphs code
        
        plt.plot(epochs_list,train_error_list,label ="training error",color='red')
        plt.plot(epochs_list,test_error_list,label='testing error',color='blue')
        plt.xlabel("Epochs")
        plt.ylabel("Men square error(mse)")
        plt.legend()
        plt.savefig('error.jpg')
        plt.close()        
        file_weights=open("weights.txt",'w')
        for neuron in self.centers_list:        
            file_weights.write(str(neuron.id)+": "+str(neuron.coeff)+"\n")   
                
        
   
# gets a 2-D list and goes vertical for each feature to do normalization        
def normalizeInputData(features_list:list):
    # run the list vertically to find min,max
    # find max in list
    max_list=list()
    min_list=list()
    for i in range (0,53):
        max_list.append(pd.DataFrame(features_list)[i].max())
        min_list.append(pd.DataFrame(features_list)[i].min())

    pos=0
    for i in range(len(features_list[0])):
        for x in features_list:
            x[i]=(x[i]-min_list[pos])/(max_list[pos]-min_list[pos])
        pos=pos+1
 
# reads the data from selwood.txt, normalizes the input data and return a list
def read_data():
    file_data=open("selwood.txt")
    input_list=list(list())
    desire_output=list()
    counter=1
    lines = file_data.readlines()
    for line in lines:
        temp_list=list()
        if(counter>3 and counter<35):
            data=line.split(",")  
            for i in range(1,55):
                num=0
                if data[i] =="<-1*":
                  num=-1
                else:
                  num=float(data[i])
                if(i==1):
                  desire_output.append(num)
                else:
                  temp_list.append(num)
            input_list.append(temp_list)
        counter=counter+1
     
    # print(desire_output)
    #print(input_list)
    
    normalizeInputData(input_list)
    
    ret_list=list()
    ret_list.append(desire_output);ret_list.append(input_list)      
    return ret_list
  
def startMethod():
    file_parameters=open("parameters.txt")
    # reading the data 
    temp=read_data()
    desire_output=temp[0]
    input_list=temp[1]
    # print(desire_output)
    # print(input_list)
    # reading The parameters of the Network
    num_centers=-1;num_inputs=-1;num_outputs=-1;learning_list=list();sigmas_list=list();iterations=-1;
    lines = file_parameters.readlines()
    for line in lines:
        data=line.split()
        parameter_name=data[0]
        value=data[1]    
        if parameter_name == 'numHiddenLayerNeurons':
            num_centers=int(value)
        elif (parameter_name == "numInputNeurons"):
            num_inputs=int(value)
        elif (parameter_name == "numOutputNeurons"):
            num_outputs=int(value)
        elif parameter_name == "learningRates":
            for i in range (1,len(data)):
                learning_list.append(float(data[i]))
        elif parameter_name == "sigmas":
            for i in range (1,len(data)):
                sigmas_list.append(float(data[i]))
        elif parameter_name == "maxIterations":
            iterations=int(value)
        else:
            print("error, incorrect data given\n")
            return     
    if num_centers==-1 or num_inputs==-1 or num_outputs==-1 or len(learning_list)!=3 or len(sigmas_list)!=num_centers or iterations==-1:
            print("Not enough data given\n")
            return 
    # print(num_centers, num_inputs,num_outputs,learning_list,sigmas_list,iterations)
    centers_pos_list=list(list())
    file_centers=open("centerVectors.txt")
    lines = file_centers.readlines()
    for line in lines:
        # data=line.split(",")
        data=line.split()
        temp_list=list()
        for num in data:
            temp_list.append(float(num))
        centers_pos_list.append(temp_list)
    #print(centers_pos_list)
    #shuffle the list
    for i in range(0,31):
        input_list[i].append(desire_output[i])
    
    shuffle(input_list)
    desire_output=list()
    for i in range(0,31):
        desire_output.append(input_list[i][53])
        input_list[i].pop(53)
        
     # now i split the data into the train and test (train: 21, test : 10)
    training_set=list();training_result=list();
    test_set=list();test_result=list();
    training_set=input_list[0:21];training_result=desire_output[0:21]
    test_set=input_list[21:31];test_result=desire_output[21:31]

    x= Network(training=training_set,training_res=training_result,testing=test_set,testing_res=test_result,center=centers_pos_list,learning_rates=learning_list,
               iters=iterations,num_of_hidden=num_centers,all_sigmas=sigmas_list)
    # x.printNetwork()   
    print("start")
    x.train_and_test()
    print("finished")

if __name__=="__main__":
    startMethod()