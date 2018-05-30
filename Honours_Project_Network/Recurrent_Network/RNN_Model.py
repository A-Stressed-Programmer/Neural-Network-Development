#Project dependancies
#--Project Imports--
#Tensorflow
import tensorflow as tf
from tensorflow.contrib import rnn
#Numpy
import numpy as np
from numpy import genfromtxt
from numpy import array
#Math
import math
#Import Plotter
import matplotlib.pyplot as plt

#Project Dependancies Imports
from File_Reader import File_Reader
filereader = File_Reader()#Declare local
from Data_Parser import Data_Parser
dataparser = Data_Parser()#Declare local
from Data_Plotter import Data_Plotter
dataplotter = Data_Plotter()#Declare local

#Main Class
class RNN():
    '''
    RNN Class housing Construction definitions for the main RNN inside Tensorflow

    Initialize main variables for RNN Construction;
    RNN(self, n_neurons, n_layers, learning_rate, fn_select, filename, epoch)
    '''
    def __init__(self, n_neurons=int, n_layers=int, learning_rate=float, fn_select="", filename="", epoch=int, batches=int, percentage=float, toggle_num=int):
        '''
        Initialize main variables for RNN Construction;
        RNN(n_neurons, n_layers, learning_rate, fn_select, filename, epoch, batches, percentage)

        n_neurons = Number of Neurons in A Layer
        n_layers = Number of Layers
        learning_rate = The learning rate of the error for gradient descent
        fn_select = Activation Function Selector
        filename = Name of File inside the "Datasets" Folder
        epoch = Total cycle count of network
        batches = Typically One for Now
        percentage = The split of Training and Testing Data
        toggle = Toggle selection for dataset switch
        '''
        #User Input feedback
        print("---Honours Neural Network Project---", "\n", "----INITIALIZE!----", "\n")
        print("VALUES: \n", "Number of Neurons:[", n_neurons,"] \n Number of Layers:[",n_layers ,"] \n Learning Rate:[", learning_rate, "] \n Function Selection:[", fn_select, "] \n Filename:[", filename,"] \n Epoch:[",epoch,"] \n Number of Batches:[",batches,"] \n")

        #--Main variables for RNN--
        #USER DEFINED VARIABLES;
        self.n_neurons = n_neurons#Neurons in each layer
        self.n_layers = n_layers#Number of Layers
        self.learning_rate = learning_rate#Learning rate for main network
        self.filename = filename#Store filename for reading
        self.epoch = epoch#Number of EPOCH runs
        self.num_batch = batches#Get number of batches  int(round(math.sqrt(self.num_data)))
        self.n_outputs =1#Neural Network Output

        #AUTOMATIC VARIABLES
        self.activation_function = RNN.get_activation_fn(fn_select)#Get Function
        self.x_axis_plotter_data, self.input_data, self.label_data = filereader.parse_data(self.filename)#GET DATA

        def auto_data_switch(toggle):
            '''
            Toggle Data switch will swap injection data from;
       
            [Open][High][Low]-->[Projection CONCAT]-->[Predictied Close] 
             to
            [Close]-->[Projection CONCAT]-->[Preicted Next Close]
            '''
            if toggle == 1:
                '''
                Toggle One, [Open][High][Low]-->[Projection CONCAT]-->[Predictied Close] 
                '''
                print("Toggle One Selected; \n [Open][High][Low]-->[Projection CONCAT]-->[Predictied Close] \n")
                #Ajust Tensor data
                self.n_inputs = 3
                #Get Data
                self.x_axis_plotter_data, open_high_low_data, closing_data = filereader.parse_data(self.filename)#GET DATA

                #Declare Local
                input_data = open_high_low_data
                self.input_data = open_high_low_data
                label_data = closing_data
                #Return Sets
                return input_data, label_data
            if toggle == 2:
                '''
                Toggle Two, [Close]-->[Projection CONCAT]-->[Preicted Next Close]
                '''
                print("Toggle Two Selected; \n [Close]-->[Projection CONCAT]-->[Preicted Next Close] \n")
                #Ajust Tensor data
                self.n_inputs = 1
                #Get Data
                self.x_axis_plotter_data, open_high_low_data, closing_data = filereader.parse_data(self.filename)#GET DATA

                #Declare Local
                input_data = closing_data[:-1]
                label_data = filereader.get_targets(closing_data)#Warning Pops First value of array structure, WILL CAUSE TENSOR VALUE ERROR
                #Return Sets
                return input_data, label_data
            else:
                '''
                Default
                Toggle Three, [Open][High][Low]-->[Projection CONCAT]-->[Predictied Close] 
                '''
                print("Toggle DEFAULT Selected; \n [Open][High][Low]-->[Projection CONCAT]-->[Predictied Close] \n")
                #Ajust Tensor data
                self.n_inputs = 3
                #Get Data
                self.x_axis_plotter_data, open_high_low_data, closing_data = filereader.parse_data(self.filename)#GET DATA

                #Declare Local
                input_data, self.input_data = open_high_low_data
                label_data = closing_data
                #Return Sets
                return input_data, label_data
                ValueError("Incorrect Toggle Select, 1-2")
        #Insert Auto Here(WARNING: MANUAL OVERRIDE OF MAIN TENSOR VALUES HERE)
        print("\n(WARNING: MANUAL OVERRIDE OF MAIN TENSOR VALUES HERE)", "\n Toggle Select:[", toggle_num, "]\n")
        t_inject_input, t_inject_label = auto_data_switch(toggle_num)

        temp_input, temp_label = dataparser.normalize_data(t_inject_input, t_inject_label, self.n_inputs)#NORMALIZE DATA
        self.train_data, self.train_label, self.test_data, self.test_label = RNN.split_data(temp_input, temp_label, percentage)#Split dataset into Train/Test

        #Save dataset to file
        print("\n Saving Training & Testing Data to files:")
        filereader.read_into_temp(self.train_data, '/Data/train_data.csv')
        filereader.read_into_temp(self.train_label, '/Data/train_label.csv')
        filereader.read_into_temp(self.test_data, '/Data/test_data.csv')
        filereader.read_into_temp(self.test_label, '/Data/test_label.csv')
        print("File Saved Successfully!\n")

        #Size Data
        self.train_size = len(self.train_data)#Train INPUT Data
        self.train_label_size = len(self.train_label)#Train LABEL Data
        self.test_size = len(self.test_data)#Test INPUT Data
        self.test_label_size = len(self.test_label)#Test LABEL Data
        #Sort Steps
        self.train_step = int(round(self.train_size/self.num_batch))
        self.test_step = int(round(self.test_size/self.num_batch))

        #Session Storage
        self.error_loss = []
        self.train_results = []
        self.test_results = []
        self.current_pos = 0#Zero position for batching
        self.t_backup = 0#Backup for [Index out of range]

        #Bi-Directional Storage
        self.train_fw_data=[]
        self.train_bw_data=[]

        #Plotter
        print("Plotter: Begining input graph data plot")
        dataplotter.input_graph(self.label_data, self.filename)
        print("Plotter: Plot Complete!")

        #Return Feedback for eval
        print("\n", "#--Network Variables--#")
        print("#--DATASET DETAILS(", filename, "): \n [Inputs(",self.n_inputs,")] \n [Outputs(", self.n_outputs, ")] \n [Size(", len(self.input_data), ")]")
        print("#--TRAINING--#", "\n Input Data Size:[",self.train_size,"]\n Number of Steps:[", self.train_step,"]\n Number of Batches:[", self.num_batch,"]")
        print("#--TESTING--#", "\n Input Data Size:[",self.test_size,"]\n Number of Steps:[", self.test_step,"]\n")
    
    def split_data(input_data, label_data, percentage):
        '''
        Split Input dataset into Training(LEFT) and Testing(RIGHT) set for RNN model in format

        split_data(input_data, label_data, percentage)

        return train_data, train_label, test_data, test_label
        '''
        #Temp Store
        data = []
        label = []
        data = input_data
        label = label_data

        #Local Variables
        #--Training
        train_data= np.array
        train_label = np.array
        #--Testing
        test_data = np.array
        test_label = np.array

        #--Training--#
        train_data = data[int(len(data) * .0) : int(len(data) * percentage)]#Return sets from 0 to Percentage
        train_label = label[int(len(label) * .0) : int(len(label) * percentage)]#Return sets from 0 to Percentage
        #--Testing--#
        test_data = data[int(len(data) * percentage) : int(len(data) * 1)]#Returns sets from Percentage to 1
        test_label = label[int(len(label) * percentage) : int(len(label) * 1)] #Returns sets from Percentage to 1
        #Return Train_data, train_label, Test_data, test_label
        return train_data, train_label, test_data, test_label

    def get_activation_fn(type):#Translate type and return activation function for network
        '''
        Select Activation functions automatically, Sigmoid(sig), Tahn(tahn), Softsign(softsign), relu(relu), default_defactor = relu
        '''
        print("Activation Select: [",type,"]")#Prompter
        activation = ''#Empty var holder
        #Main Loop

        if type == 'sig':#SIGMOID
            #Return Sigmoid
            '''
            f(x)=1/1+e^-x
            '''
            print("Sigmoid: [f(x)=1/1+e^-x] Selected!")
            activation = tf.nn.softmax

        elif type == 'tahn':#TAHN
            #Return Tahn
            '''
            f(x)=tanh(x)=e^x - e^-x/e^x + e^-x
            '''
            print("Tahn: [f(x)=tanh(x)=e^x - e^-x/e^x + e^-x] Selected!")
            activation = tf.tanh

        elif type == 'softsign':#SOFTSIGN
            #Return Softsign
            '''
            f(x)=x/1+[X]
            '''
            print("Softsign Selected!")
            activation = tf.nn.softsign

        elif type == 'relu':#RECTIFIED LINEAR UNIT
            #Return Relu
            '''
            Rectified Linear Unit(ReLU)
            f(x)={0 for x < 0
                 {x for x >/ 0
            '''
            print("Rectified Linear Unit(ReLU) Selected!")
            activation = tf.nn.relu

        else:#EXCEPTION
            #Return default
            print("DEFAULT SETTINGS!")#CHANGE ME
            activation = tf.nn.selu

        return activation#Return Activation Function
        print("Activation Retured to main task!")#Conformation of task completion
        #--WARNING ONLY RELU WORKS DUE TO UN-NORMALIZED DATA--#

    #--------------------------#
    #Recurrent Neural Networks#
    #-Upgrade Complete
    #-Saver Corrected
    #-Operational

    def train_rnn_model(self):
        '''
        Train Deep RNN Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.train_data#Data
        dataset_y = self.train_label#Label
        steps = self.train_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        prob = 0.5#Keep Prop for Outputs of Neural Network
        num_data = self.train_size#Size of Network

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Local Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > num_data:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

                #OLD
                self.t_backup = self.t_backup * 0#Reset
                self.t_backup = self.current_pos#Setnew

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            y_batch = array(y_batch).reshape(steps,num_out)
            #Return Batching
            return x_batch, y_batch

        #Placeholders#
        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER
        Y = tf.placeholder(tf.float32, shape=[steps, num_out], name="y")#TARGET_BATCH_DATA_PLACEHOLDER

        #Main RNN Framework#
        rnn_cell=[]
        for _ in range(self.n_layers):
            cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)#Layer declare
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=prob, state_keep_prob=prob)#Dropper layer
            rnn_cell.append(cell)#Append to rnn_cell

        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(rnn_cell)#Append cell to layers, Stack cells to layers
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)#Declare outputs and states for loss

        #OUTPUT PROJECT WRAPPER#
        with tf.name_scope("wrapper"):
            '''
            Output Projection Wrapper
            '''
            stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            outputs = tf.reshape(stacked_outputs, [steps, num_out])

        #LOSS#
        with tf.name_scope("loss"):
            loss = tf.reduce_sum(tf.square(outputs - Y))
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            training_op = optimizer.minimize(loss)

        #Initialize Variables
        init = tf.global_variables_initializer()
        #Create Saver
        saver = tf.train.Saver()

        #--Main Session--#
        with tf.Session() as sess:
            init.run()#Initialize
            #---Outer Epoch Shell---#
            print("---Training RNN Model---")
            for epoch  in range(self.epoch):
                #---Inner Iteration shell----#
                self.train_results = []*0#Zero Results to grab last epoch(Most Accurate)
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch, y_batch = next_batch()#Train           
                    _, output_val, mse = sess.run([training_op, outputs, loss], feed_dict={X: x_batch, Y: y_batch})#TRAINING
                    results_data = []#Append data to array
                    for i in range(len(output_val)):
                        results_data.append(output_val[i])
                    self.train_results.append(results_data)#Append Results
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Training RNN: Epoch: [",epoch ,"/" , self.epoch,"] ", "MSE: [",mse,"]")#Prompter ,"Output: [", output_val, "]"
                error = str(mse)
                self.error_loss.append(error)
            #---Outer Shell End---#
            print("---Training Completed!---")
            print("Saving!")
            saver.save(sess, "Recurrent_Network/Saves/RNN/Recurrent_Neural_Network.ckpt") 

            #File Printers#
            print("Printing to Files")
            filereader.read_into_temp(np.ravel(self.train_results), "/RNN/train_results.csv")#Save to file
            filereader.read_into_temp(self.error_loss, "/RNN/error_results.csv")#Save to File
            print("Task Completed!")

            #Plotters#
            print("Plotting...")
            dataplotter.train_graph(self.filename,np.ravel(self.train_results), 'rnn')
            dataplotter.plt_error_graph(self.filename, np.ravel(self.error_loss), self.epoch, 'rnn')
            dataplotter.c_train_graph(self.filename,np.ravel(self.train_results), np.ravel(self.train_label), 'rnn')
            print("Plot Completed!")
            #---END OF TRAIN--#

    def test_rnn_model(self):
        '''
        Train Basic RNN Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset TensorFlow information

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.test_data#Data
        dataset_y = self.test_label#Label
        steps = self.test_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        num_data = self.test_size

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > self.test_size:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            #Return Batching
            return x_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER

        '''
        #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
        cells = [
            tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]
        '''

        #Declare Cell
        rnn_cell=[]
        for _ in range(self.n_layers):
            #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
            cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            #Dropper layer for over saturated data
            rnn_cell.append(cell)

        #Stack cells to layers
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(rnn_cell)#Append cell to layers
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)#Declare outputs and states for loss

        #OUTPUT PROJECT WRAPPER--For input Data
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
        outputs = tf.reshape(stacked_outputs, [steps, num_out])

        #Create Saver
        saver = tf.train.Saver()

        #--Main Session--#
        with tf.Session() as sess:
            saver.restore(sess, "Recurrent_Network/Saves/RNN/Recurrent_Neural_Network.ckpt")#Restore Files
            #---Outer Epoch Shell---#
            print("---Testing RNN Model---")
            for epoch  in range(1):
                #---Inner Iteration shell----#
                self.test_results = []*0#Zero 
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch = next_batch()
                    output_val = sess.run([outputs], feed_dict={X: x_batch})
                    self.test_results.append(output_val)
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Testing: Epoch: [",epoch ,"/" , self.epoch,"] ")#Prompter ,"Output: [", output_val, "]"", "MSE: [",mse,"]"
            #---Outer Shell End---#
            print("---Testing Completed!---")

            #File Printer
            filereader.read_into_temp(np.ravel(self.test_results), "/RNN/test_results.csv")#Print Into File

            #Numpy.Ravel
            training_results = np.ravel(self.train_results)
            testing_results = np.ravel(self.test_results)

            #Plotters
            print("Plotting...")
            dataplotter.test_graph(self.filename,testing_results, 'rnn')
            dataplotter.c_test_graph(self.filename,testing_results, np.ravel(self.test_label), 'rnn')
            dataplotter.c_combined_graph(self.filename, self.train_label, self.test_label,np.ravel(self.train_results), np.ravel(self.test_results), 'rnn')
            print("Plot Completed!")

            print("\n #---End of Recurrent Neural Network---# \n")
            #---END OF TEST--#

    #--------------------------#
    #Bi-Directional Recurrent Neural Networks# #INCOMPLETE#

    def train_bi_rnn_model(self):
        '''
        Train Deep Bi-Directional RNN Cell model, Recurrance within the main cell body
        '''
        print("#---Bi-Directional RNN (BRNN)---#")#Identifier

        tf.reset_default_graph()#Reset
        self.error_loss = []*0#Reset Error

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.train_data#Data
        dataset_y = self.train_label#Label
        steps = self.train_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        prob = 1.0#Keep Prop for Outputs of Neural Network
        num_data = self.train_size#Size of Network

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Local Variables
            x_batch = []
            y_batch = []
            z_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > num_data:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

                #OLD
                self.t_backup = self.t_backup * 0#Reset
                self.t_backup = self.current_pos#Setnew

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            y_batch = array(y_batch).reshape(steps,num_out)
            #Return Batching
            return x_batch, y_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER
        Y = tf.placeholder(tf.float32, shape=[steps, num_out], name="y")#TARGET_BATCH_DATA_PLACEHOLDER
       
        #Forward cell
        fw_cells = [
            tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]
        #Append Dropout Wrappper
        fw_cells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob) for cell in fw_cells]
        #Append to multicell
        fw_multi_layer_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)

        #Backward Cell
        bw_cells = [
            tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]
        #Append Dropout Wrapper
        bw_cells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob) for cell in bw_cells]
        #Append to multicell
        bw_multi_layer_cell = tf.contrib.rnn.MultiRNNCell(bw_cells)
        
        rnn_outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_multi_layer_cell, cell_bw=bw_multi_layer_cell, dtype=tf.float32, inputs=X)

        #Break Tuple, Does not work on Stack_Bidirectional due to "Tensor Mismatch error"
        fw_outputs, bw_outputs = rnn_outputs
        fw_states, bw_states = states

        #Backward Projection Wrapper
        with tf.name_scope("bw_wrapper"):
            stacked_rnn_outputs = tf.reshape(bw_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            back_outputs = tf.reshape(stacked_outputs, [steps, num_out])
        #Forward Projection Wrapper
        with tf.name_scope("fw_wrapper"):
            stacked_rnn_outputs = tf.reshape(fw_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            forw_outputs = tf.reshape(stacked_outputs, [steps, num_out])
        
        #Total Loss
        with tf.name_scope("total_loss"):
            total_loss = tf.reduce_sum(tf.square(rnn_outputs - Y))#Sum of Loss
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)#OPTIMIZER
            total_training_op = optimizer.minimize(total_loss)
        #Forward_LOSS
        with tf.name_scope("fw_loss"):
            fw_loss = tf.reduce_sum(tf.square(fw_outputs - Y))
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            fw_training_op = optimizer.minimize(fw_loss)
        #Backward_LOSS/INVERT ME!
        with tf.name_scope("bw_loss"):
            bw_loss = tf.reduce_sum(tf.square(bw_outputs - Y))
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            bw_training_op = optimizer.minimize(bw_loss)

        init = tf.global_variables_initializer()#Initialize
        saver = tf.train.Saver()#Save Me

        #Local Non Reset arrays
        fw_loss_out=[]#Forward Loss
        bw_loss_out=[]#Backward Loss

        #--Main Session--#
        with tf.Session() as sess:
            init.run()#Initialize
            #---Outer Epoch Shell---#
            print("---Training Bi-Directional RNN Model---")
            for epoch  in range(self.epoch):
                #---Inner Iteration shell----#
                fw_results=[]#Forward Results
                bw_results=[]#Backward Results
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch, y_batch = next_batch()#Train           
                    #I'm new and shiney
                    _, _, _, fw_output_vals, bw_output_vals, fw_mse, bw_mse = sess.run([total_training_op, fw_training_op, bw_training_op, forw_outputs, back_outputs, fw_loss, bw_loss], feed_dict={X: x_batch, Y: y_batch})
                    #Append Results
                    fw_results.append(fw_output_vals)#Append Forward
                    bw_results.append(bw_output_vals)#Append Backward
                    fw_loss_out.append(str(fw_mse))#Append forward Loss
                    bw_loss_out.append(str(bw_mse))#Append Backward Loss
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Training Bi-Directional RNN: Epoch: [",epoch ,"/" , self.epoch,"] ", "Forward_MSE: [",fw_mse,"] ", "Backward_MSE: [",bw_mse, "]")#Prompter ,"Output: [", output_val, "]"
                #String Error, Matplotlib does not like tensors
            #---Outer Shell End---#
            print("---Training Completed!---")

            #Saving
            print("Saving!")
            saver.save(sess, "Recurrent_Network/Saves/BI-RNN/Bi-Direcitonal-RNN.ckpt")
            print("Save Complete! \n")

            #Printers
            print("Printing to Files")
            filereader.read_into_temp(np.ravel(fw_results), "/BI-RNN/forward_train.csv")#Save to file
            filereader.read_into_temp(np.ravel(bw_results), "/BI-RNN/backward_train.csv")#Save to file
            filereader.read_into_temp(np.ravel(fw_loss_out), "/BI-RNN/forward_error.csv")#Save to file
            filereader.read_into_temp(np.ravel(bw_loss_out), "/BI-RNN/backward_error.csv")#Save to file
            filereader.read_into_temp(self.error_loss, "/BI-RNN/error_results.csv")#Save to File
            print("Task Completed! \n")

            #Plotters
            print("Plotting...")
            dataplotter.bi_train_graph(self.filename, np.ravel(fw_results), np.ravel(bw_results), 'bi-rnn')
            dataplotter.plt_bi_error_graph(self.filename, np.ravel(fw_loss_out), np.ravel(bw_loss_out), self.epoch, 'bi-rnn')
            dataplotter.bi_c_train_graph(self.filename, np.ravel(fw_results), np.ravel(bw_results), self.train_label, 'bi-rnn')
            print("Plot Completed! \n")

            #Save for pass forward to Tester
            self.train_fw_data.append(fw_results)
            self.train_bw_data.append(bw_results)
            print("Training Bi-Directional RNN End")
            #---END OF TRAIN--#

    def test_bi_rnn_model(self):
        '''
        Train Bi-Directional RNN Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.test_data#Data
        steps = self.test_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        num_data = self.test_size

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > self.test_size:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            #Return Batching
            return x_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER

        #Forward cell
        fw_cells = [
            tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]
        #Append to multicell
        fw_multi_layer_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)

        #Backward Cell
        bw_cells = [
            tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]
        #Append to multicell
        bw_multi_layer_cell = tf.contrib.rnn.MultiRNNCell(bw_cells)
        
        rnn_outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_multi_layer_cell, cell_bw=bw_multi_layer_cell, dtype=tf.float32, inputs=X)

        #Break Tuple
        fw_outputs, bw_outputs = rnn_outputs
        fw_states, bw_states = states

        #Backward Projection Wrapper
        with tf.name_scope("bw_wrapper"):
            stacked_rnn_outputs = tf.reshape(bw_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            back_outputs = tf.reshape(stacked_outputs, [steps, num_out])
        #Forward Projection Wrapper
        with tf.name_scope("fw_wrapper"):
            stacked_rnn_outputs = tf.reshape(fw_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            forw_outputs = tf.reshape(stacked_outputs, [steps, num_out])

        saver = tf.train.Saver()#Save Me   

        #--Main Session--#
        with tf.Session() as sess:
            saver.restore(sess, "Recurrent_Network/Saves/BI-RNN/Bi-Direcitonal-RNN.ckpt")#Restore Files
            #---Outer Epoch Shell---#
            print("\n---Testing Bi-Directional RNN Model---")
            for epoch  in range(1):
                #---Inner Iteration shell----#
                fw_results = []
                bw_results = []
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch = next_batch()
                    #New Testing Layout
                    fw_output_vals, bw_output_vals = sess.run([forw_outputs, back_outputs], feed_dict={X: x_batch})
                    #Append Results
                    fw_results.append(array(fw_output_vals).reshape(steps,num_out))
                    bw_results.append(array(bw_output_vals).reshape(steps,num_out))
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Testing LSTM: Epoch: [",epoch ,"/" , self.epoch,"] ")#Prompter
            #---Outer Shell End---#
            print("---Testing Completed!---\n")

            #Printers
            print("Printing to Files")
            filereader.read_into_temp(fw_results, "/BI-RNN/forward_test.csv")#Save to file
            filereader.read_into_temp(bw_results, "/BI-RNN/backward_test.csv")#Save to file
            print("Plot Completed! \n")

            #Plotters
            print("Plotting...")
            dataplotter.bi_test_graph(self.filename, np.ravel(fw_results), np.ravel(bw_results), 'bi-rnn')
            dataplotter.bi_c_test_graph(self.filename, np.ravel(fw_results), np.ravel(bw_results), self.test_label, 'bi-rnn')
            dataplotter.bi_combined_graph(self.filename, np.ravel(self.train_fw_data), np.ravel(self.train_bw_data), np.ravel(fw_results), np.ravel(bw_results), self.train_label, self.test_label,  'bi-rnn')
            print("Plot Completed!")

            print("---Test Bi-Directional RNN END---")
            #---END OF TEST--#

    #--------------------------# #UPDATE#
    #Long Short-Term Memory#
    #-Saver Corrected
    #-Operational

    def train_lstm_model(self):
        '''
        Train Deep LSTM Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset
        self.error_loss = []*0#Reset Error

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.train_data#Data
        dataset_y = self.train_label#Label
        steps = self.train_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        prob = 1.0#Keep Prop for Outputs of Neural Network
        num_data = self.train_size#Size of Network

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Local Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > num_data:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

                #OLD
                self.t_backup = self.t_backup * 0#Reset
                self.t_backup = self.current_pos#Setnew

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            y_batch = array(y_batch).reshape(steps,num_out)
            #Return Batching
            return x_batch, y_batch


        #PLACEHOLDER#
        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER
        Y = tf.placeholder(tf.float32, shape=[steps, num_out], name="y")#TARGET_BATCH_DATA_PLACEHOLDER
        
        #Main LSTM Framework#
        lstm_cell=[]
        for _ in range(self.n_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, forget_bias=1.0, activation=self.activation_function)#Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob)#Dropper layer for over saturated data
            lstm_cell.append(cell)#Append to rnn_cell

        #Stack cells to layers
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(lstm_cell)
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

        #OUTPUT PROJECT WRAPPER--For input Data
        with tf.name_scope("wrapper"):
            stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            outputs = tf.reshape(stacked_outputs, [steps, num_out])

        #LOSS
        with tf.name_scope("loss"):
            loss = tf.reduce_sum(tf.square(outputs - Y))#Sum of Loss
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)#OPTIMIZER, ADAM is MOST ACCURATE MODEL
            training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        #--Main Session--#
        with tf.Session() as sess:
            init.run()#Initialize
            #---Outer Epoch Shell---#
            print("---Training LSTM Model---")
            for epoch  in range(self.epoch):
                #---Inner Iteration shell----#
                self.train_results = []*0#Zero Results to grab last epoch(Most Accurate)
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch, y_batch = next_batch()#Train           
                    _, output_val, mse = sess.run([training_op, outputs, loss], feed_dict={X: x_batch, Y: y_batch})#TRAINING
                    results_data = []#Append data to array
                    for i in range(len(output_val)):
                        results_data.append(output_val[i])
                    self.train_results.append(results_data)#Append Results
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Training LSTM: Epoch: [",epoch ,"/" , self.epoch,"] ", "MSE: [",mse,"]")#Prompter ,"Output: [", output_val, "]"
                error = str(mse)
                self.error_loss.append(error)
            #---Outer Shell End---#
            print("---Training Completed!---")

            print("Saving!")
            saver.save(sess, "Recurrent_Network/Saves/LSTM/LSTM_Network.ckpt")

            print("Printing to Files")
            filereader.read_into_temp(np.ravel(self.train_results), "/LSTM/train_results.csv")#Save to file
            filereader.read_into_temp(self.error_loss, "/LSTM/error_results.csv")#Save to File
            print("Task Completed!")

            #Plotters
            print("Plotting...")
            dataplotter.train_graph(self.filename,np.ravel(self.train_results), 'lstm')
            dataplotter.plt_error_graph(self.filename, np.ravel(self.error_loss), self.epoch, 'lstm')
            dataplotter.c_train_graph(self.filename,np.ravel(self.train_results), np.ravel(self.train_label), 'lstm')
            print("Plot Completed!")
            #---END OF TRAIN--#

    def test_lstm_model(self):
        '''
        Train Basic RNN Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.test_data#Data
        dataset_y = self.test_label#Label
        steps = self.test_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        num_data = self.test_size

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > self.test_size:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            #Return Batching
            return x_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER

        '''
        #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
        cells = [
            tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]
        '''

        #Main LSTM Framework#
        lstm_cell=[]
        for _ in range(self.n_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, forget_bias=1.0, activation=self.activation_function)#Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
            lstm_cell.append(cell)#Append to rnn_cell

        #Stack cells to layers
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(lstm_cell)#Append cell to layers
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)#Declare outputs and states for loss

        #OUTPUT PROJECT WRAPPER--For input Data
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
        outputs = tf.reshape(stacked_outputs, [steps, num_out])

        #Create Saver
        saver = tf.train.Saver()

        #--Main Session--#
        with tf.Session() as sess:
            saver.restore(sess, "Recurrent_Network/Saves/LSTM/LSTM_Network.ckpt")#Restore Files
            #---Outer Epoch Shell---#
            print("---Testing LSTM Model---")
            for epoch  in range(1):
                #---Inner Iteration shell----#
                self.test_results = []*0#Zero 
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch = next_batch()
                    output_val = sess.run([outputs], feed_dict={X: x_batch})
                    self.test_results.append(output_val)
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Testing LSTM: Epoch: [",epoch ,"/" , self.epoch,"] ")#Prompter ,"Output: [", output_val, "]"", "MSE: [",mse,"]"
            #---Outer Shell End---#
            print("---Testing Completed!---")
            filereader.read_into_temp(np.ravel(self.test_results), "/LSTM/test_results.csv")#Print Into File

            #Numpy.Ravel
            training_results = np.ravel(self.train_results)
            testing_results = np.ravel(self.test_results)

            #Plotters
            print("Plotting...")
            dataplotter.test_graph(self.filename,testing_results, 'lstm')
            dataplotter.c_test_graph(self.filename,testing_results, np.ravel(self.test_label), 'lstm')
            dataplotter.c_combined_graph(self.filename, self.train_label, self.test_label,np.ravel(self.train_results), np.ravel(self.test_results), 'lstm')
            print("Plot Completed!")

            print("---LSTM END---")
            #---END OF TEST--#

    #--------------------------#
    #Bi-Directional Long Short-Term Memory# #INCOMPLETE#

    def train_bi_lstm_model(self):
        '''
        Train Deep Bi-Directional LSTM Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset
        self.error_loss = []*0#Reset Error

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.train_data#Data
        dataset_y = self.train_label#Label
        steps = self.train_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        prob = 1.0#Keep Prop for Outputs of Neural Network
        num_data = self.train_size#Size of Network

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Local Variables
            x_batch = []
            y_batch = []
            z_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > num_data:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

                #OLD
                self.t_backup = self.t_backup * 0#Reset
                self.t_backup = self.current_pos#Setnew

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            y_batch = array(y_batch).reshape(steps,num_out)
            #Return Batching
            return x_batch, y_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER
        Y = tf.placeholder(tf.float32, shape=[steps, num_out], name="y")#TARGET_BATCH_DATA_PLACEHOLDER
       
        #Forward cell
        fw_cells = [
            tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, activation=self.activation_function, forget_bias=1.0)
            for layer in range(self.n_layers)]
        #Append Dropout Wrappper
        fw_cells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob) for cell in fw_cells]
        #Append to multicell
        fw_multi_layer_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)

        #Backward Cell
        bw_cells = [
            tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, activation=self.activation_function, forget_bias=1.0)
            for layer in range(self.n_layers)]
        #Append Dropout Wrapper
        bw_cells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob) for cell in bw_cells]
        #Append to multicell
        bw_multi_layer_cell = tf.contrib.rnn.MultiRNNCell(bw_cells)
        
        rnn_outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_multi_layer_cell, cell_bw=bw_multi_layer_cell, dtype=tf.float32, inputs=X)

        #Break Tuple, Does not work on Stack_Bidirectional due to "Tensor Mismatch error"
        fw_outputs, bw_outputs = rnn_outputs
        fw_states, bw_states = states

        #Backward Projection Wrapper
        with tf.name_scope("bw_wrapper"):
            stacked_rnn_outputs = tf.reshape(bw_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            back_outputs = tf.reshape(stacked_outputs, [steps, num_out])
        #Forward Projection Wrapper
        with tf.name_scope("fw_wrapper"):
            stacked_rnn_outputs = tf.reshape(fw_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            forw_outputs = tf.reshape(stacked_outputs, [steps, num_out])
        
        #Total Loss
        with tf.name_scope("total_loss"):
            total_loss = tf.reduce_sum(tf.square(rnn_outputs - Y))#Sum of Loss
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)#OPTIMIZER
            total_training_op = optimizer.minimize(total_loss)
        #Forward_LOSS
        with tf.name_scope("fw_loss"):
            fw_loss = tf.reduce_sum(tf.square(fw_outputs - Y))
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            fw_training_op = optimizer.minimize(fw_loss)
        #Backward_LOSS/INVERT ME!
        with tf.name_scope("bw_loss"):
            bw_loss = tf.reduce_sum(tf.square(bw_outputs - Y))
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            bw_training_op = optimizer.minimize(bw_loss)

        init = tf.global_variables_initializer()#Initialize
        saver = tf.train.Saver()#Save Me

        #Local Non Reset arrays
        fw_loss_out=[]#Forward Loss
        bw_loss_out=[]#Backward Loss

        #--Main Session--#
        with tf.Session() as sess:
            init.run()#Initialize
            #---Outer Epoch Shell---#
            print("---Training Bi-Directional LSTM Model---")
            for epoch  in range(self.epoch):
                #---Inner Iteration shell----#
                fw_results=[]#Forward Results
                bw_results=[]#Backward Results
                for iteration in range(self.num_batch):
                    '''
                    Reworking;
                    Fw_Loss caluclation
                    bw_Loss Caluclation
                    fw_results
                    bw_results
                    fw_training
                    bw_training
                    '''
                    #---Execution Shell---#
                    x_batch, y_batch = next_batch()#Train           
                    #I'm new and shiney
                    _, _, _, fw_output_vals, bw_output_vals, fw_mse, bw_mse = sess.run([total_training_op, fw_training_op, bw_training_op, forw_outputs, back_outputs, fw_loss, bw_loss], feed_dict={X: x_batch, Y: y_batch})
                    #Append Results
                    fw_results.append(fw_output_vals)#Append Forward
                    bw_results.append(bw_output_vals)#Append Backward
                    fw_loss_out.append(str(fw_mse))#Append forward Loss
                    bw_loss_out.append(str(bw_mse))#Append Backward Loss
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Training Bi-Directional LSTM: Epoch: [",epoch ,"/" , self.epoch,"] ", "Forward_MSE: [",fw_mse,"] ", "Backward_MSE: [",bw_mse, "]")#Prompter ,"Output: [", output_val, "]"
                #String Error, Matplotlib does not like tensors
            #---Outer Shell End---#
            print("---Training Completed!---")

            #Saving
            print("Saving!")
            saver.save(sess, "Recurrent_Network/Saves/BI-LSTM/Bi-Direcitonal-LSTM.ckpt")
            print("Save Complete! \n")

            #Printers
            print("Printing to Files")
            filereader.read_into_temp(np.ravel(fw_results), "/BI-LSTM/forward_train.csv")#Save to file
            filereader.read_into_temp(np.ravel(bw_results), "/BI-LSTM/backward_train.csv")#Save to file
            filereader.read_into_temp(np.ravel(fw_loss_out), "/BI-LSTM/forward_error.csv")#Save to file
            filereader.read_into_temp(np.ravel(bw_loss_out), "/BI-LSTM/backward_error.csv")#Save to file
            filereader.read_into_temp(self.error_loss, "/BI-LSTM/error_results.csv")#Save to File
            print("Task Completed! \n")

            #Plotters
            print("Plotting...")
            dataplotter.bi_train_graph(self.filename, np.ravel(fw_results), np.ravel(bw_results), 'bi-lstm')
            dataplotter.plt_bi_error_graph(self.filename, np.ravel(fw_loss_out), np.ravel(bw_loss_out), self.epoch, 'bi-lstm')
            dataplotter.bi_c_train_graph(self.filename, np.ravel(fw_results), np.ravel(bw_results), self.train_label, 'bi-lstm')
            print("Plot Completed! \n")

            #Save for pass forward to Tester
            self.train_fw_data.append(fw_results)
            self.train_bw_data.append(bw_results)
            #---END OF TRAIN--#

    def test_bi_lstm_model(self):
        '''
        Train Basic RNN Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.test_data#Data
        steps = self.test_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        num_data = self.test_size

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > self.test_size:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            #Return Batching
            return x_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER

        #Forward cell
        fw_cells = [
            tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, activation=self.activation_function, forget_bias=1.0)
            for layer in range(self.n_layers)]
        #Append to multicell
        fw_multi_layer_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)

        #Backward Cell
        bw_cells = [
            tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, activation=self.activation_function, forget_bias=1.0)
            for layer in range(self.n_layers)]
        #Append to multicell
        bw_multi_layer_cell = tf.contrib.rnn.MultiRNNCell(bw_cells)
        
        rnn_outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_multi_layer_cell, cell_bw=bw_multi_layer_cell, dtype=tf.float32, inputs=X)

        #Break Tuple, Does not work on Stack_Bidirectional due to "Tensor Mismatch error"
        fw_outputs, bw_outputs = rnn_outputs#Why rnn_outputs = steps*2=tf.concat(fw_data, bw_data)=steps*2, but why still in output?
        fw_states, bw_states = states

        #Backward Projection Wrapper
        with tf.name_scope("bw_wrapper"):
            stacked_rnn_outputs = tf.reshape(bw_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            back_outputs = tf.reshape(stacked_outputs, [steps, num_out])
        #Forward Projection Wrapper
        with tf.name_scope("fw_wrapper"):
            stacked_rnn_outputs = tf.reshape(fw_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            forw_outputs = tf.reshape(stacked_outputs, [steps, num_out])

        saver = tf.train.Saver()#Save Me   

        #--Main Session--#
        with tf.Session() as sess:
            saver.restore(sess, "Recurrent_Network/Saves/BI-LSTM/Bi-Direcitonal-LSTM.ckpt")#Restore Files
            #---Outer Epoch Shell---#
            print("\n---Testing Bi-Directional LSTM Model---")
            for epoch  in range(1):
                #---Inner Iteration shell----#
                fw_results = []
                bw_results = []
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch = next_batch()
                    #New Testing Layout
                    fw_output_vals, bw_output_vals = sess.run([forw_outputs, back_outputs], feed_dict={X: x_batch})
                    #Append Results
                    fw_results.append(array(fw_output_vals).reshape(steps,num_out))
                    bw_results.append(array(bw_output_vals).reshape(steps,num_out))
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Testing LSTM: Epoch: [",epoch ,"/" , self.epoch,"] ")#Prompter
            #---Outer Shell End---#
            print("---Testing Completed!---\n")

            #Printers
            print("Printing to Files")
            filereader.read_into_temp(fw_results, "/BI-LSTM/forward_test.csv")#Save to file
            filereader.read_into_temp(bw_results, "/BI-LSTM/backward_test.csv")#Save to file
            print("Plot Completed! \n")

            #Plotters
            print("Plotting...")
            dataplotter.bi_test_graph(self.filename, np.ravel(fw_results), np.ravel(bw_results), 'bi-lstm')
            dataplotter.bi_c_test_graph(self.filename, np.ravel(fw_results), np.ravel(bw_results), self.test_label, 'bi-lstm')
            dataplotter.bi_combined_graph(self.filename, np.ravel(self.train_fw_data), np.ravel(self.train_bw_data), np.ravel(fw_results), np.ravel(bw_results), self.train_label, self.test_label,  'bi-lstm')
            print("Plot Completed!")

            print("---LSTM END---")
            #---END OF TEST--#

    #--------------------------#

    def plot_concat_graphs(self):
        '''
        Plots and stores nn_error, nn_train, nn_test graphs 
        '''
        print("Reading From Files")
        train_label_data = filereader.read_csv('Recurrent_Network/Temp/Data/train_label.csv')
        test_label_data = filereader.read_csv('Recurrent_Network/Temp/Data/test_label.csv')

        rnn_train_data = filereader.read_csv('Recurrent_Network/Temp/RNN/train_results.csv')
        rnn_test_data = filereader.read_csv('Recurrent_Network/Temp/RNN/test_results.csv')

        lstm_train_data = filereader.read_csv('Recurrent_Network/Temp/LSTM/train_results.csv')
        lstm_test_data = filereader.read_csv('Recurrent_Network/Temp/LSTM/train_results.csv')
        print("File Read Complete!")

        dataplotter.plt_nn_train(train_label_data, np.ravel(rnn_train_data), np.ravel(lstm_train_data))
        dataplotter.plt_nn_test(test_label_data, np.ravel(rnn_test_data), np.ravel(lstm_test_data))


        def gradientDescent():
            learning_rate = 0.05

            for value in range(number_data_points):
                x = Input_Data[i].value
                y = Expected_Data[i].value
                
                prediction = M * x + c
                error = y - prediction
                
                M = M + error * x * learning_rate
                c = c + error * learning_rate
            
       