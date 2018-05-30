#Project Imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Data_Plotter():
    '''
    Plotter houses main MatPlotLib.Extenstion(PythonPlot) commands to visualize datasets

    Initialize main variables for Plotter;
    RNN(self, n_neurons, n_layers, learning_rate, fn_select, filename, epoch)
    '''

    def model_selector(nn_type, graph_sel):
        '''
        Take user selector return file user data
        '''

        save_file = ''

        #SPLIT RNN/LSTM
        if graph_sel == 'single-train':
            save_file = "Recurrent_Network/Graphs/" + nn_type.upper() + "/Single/train_data.png"
            #Return details for plotters
            return nn_type.upper(), save_file
        if graph_sel == 'single-test':
            save_file = "Recurrent_Network/Graphs/" + nn_type.upper() + "/Single/test_data.png"
            #Return details for plotters
            return nn_type.upper(), save_file
        if graph_sel == 'error':
            save_file = "Recurrent_Network/Graphs/" + nn_type.upper() + "/Error/train_error.png"
            #Return details for plotters
            return nn_type.upper(), save_file
        if graph_sel == 'compare-train':
            save_file = "Recurrent_Network/Graphs/" + nn_type.upper() + "/Compare/compare_train.png"
            #Return details for plotters
            return nn_type.upper(), save_file
        if graph_sel == 'compare-test':
            save_file = "Recurrent_Network/Graphs/" + nn_type.upper() + "/Compare/compare_test.png"
            #Return details for plotters
            return nn_type.upper(), save_file
        if graph_sel == 'combined':
            save_file = "Recurrent_Network/Graphs/" + nn_type.upper() + "/Compare/combined.png"
            #Return details for plotters
            return nn_type.upper(), save_file
        else:
            print("ERROR: Model selector incorrectly assigned!")
            print(save_file)

#--------------------------------------------------------------------------#

    def input_graph(self,input_data, filename):
        '''
        Plot Input data graph from inputted datasets and append to local storage;

        input_graph(self, input_data, filename):
        '''
        while True:
            try:#Try/Catch for error Running
                print("Running [Input_Graph()]!")
                title = ("Initial Input Data:[" + filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(£)")
                plt.title(title,color="red", fontsize=20)#Plot Title

                plt.plot(input_data,linestyle='-',linewidth=2,color="red" ,label="Label Data")#Plot Data
                plt.legend(loc="upper left")#Lengend
                plt.savefig("Recurrent_Network/Graphs/input_data.png")#Save me
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Input_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return 

#--------------------------------------------------------------------------#

    def train_graph(self,filename, output_data, nn_type):
        '''
        Plot Training data graph from network output data and append to local storage;

        train_graph(self,filename, output_data, nn_type):
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type, 'single-train')

        while True:
            try:#Try/Catch for error Running
                print("Running [Train_Graph()]!")
                title = ("Training("+ information +"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(£)")
                plt.title(title, color="blue", fontsize=14)
                plt.plot(output_data, linestyle='-',linewidth=2, color="blue", label="Predicted Data")#Plot Predicted
                plt.legend(loc="upper left")
                plt.savefig(save_file)#Save to File
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Train_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return DONE

    def test_graph(self,filename, output_data, nn_type):
        '''
        Plot Testing data graph from network output data and label data and append to local storage

        test_graph(self, output_data, label_data):
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type, 'single-test')

        while True:
            try:#Try/Catch for error Running
                print("Running [test_graph()]!")
                title = ("Testing("+information+"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.title(title, color="green", fontsize=14)
                plt.plot(output_data, linestyle='-',linewidth=2, color="green", label="Predicted Data")#Plot Predicted
                plt.legend(loc="upper left")
                plt.savefig(save_file)
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Test_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return DONE

#--------------------------------------------------------------------------#

    def c_train_graph(self,filename, output_data, label_data, nn_type):
        '''
        Plot Training data graph from network output data and label data and append to local storage;

        train_graph(self,filename, output_data, label_data):
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type, 'compare-train')

        while True:
            try:#Try/Catch for error Running
                print("Running [c_train_graph()]!")
                title = ("Training Compare("+information+"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.title(title, color="blue", fontsize=14)
                plt.plot(output_data, linestyle='-',linewidth=1, color="blue", label="Training Prediction")#Plot Predicted
                plt.plot(label_data, linestyle='-',linewidth=1, color="red", label="Actual Data")#Plot Actual
                plt.legend(loc="upper left")
                plt.savefig(save_file)
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Train_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return

    def c_test_graph(self,filename, output_data, label_data, nn_type):
        '''
        Plot Testing data graph from network output data and label data and append to local storage;

        c_test_graph(self, output_data, label_data):
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type, 'compare-test')

        while True:
            try:#Try/Catch for error Running
                plt.close()
                print("Running [c_test_graph()]!")
                title = ("Testing Compare("+information+"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.title(title, color="blue", fontsize=14)
                plt.plot(output_data, linestyle='-',linewidth=1, color="blue", label="Training Prediction")#Plot Predicted
                plt.plot(label_data, linestyle='-',linewidth=1, color="red", label="Actual Data")#Plot Actual
                plt.legend(loc="upper left")
                plt.savefig(save_file)
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Train_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return

    def c_combined_graph(self,filename, train_label, test_label, train_data, test_data, nn_type):
        '''
        Compare Training and Testing Data;
        c_combined_graph(self,filename, train_label, test_label, train_data, test_data)
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type, 'combined')

        while True:
            try:#Try/Catch for error Running
                print("Begining [c_combined_graph] Plotting")
                #Top Plot-Training
                plt.subplot(2,1,1)
                title = ("Training("+information+"):["+ filename + "]")
                plt.plot(train_label, linestyle='-',linewidth=1, color="g", label="Train Label", alpha=0.5)#Plot Acutal
                plt.plot(train_data, linestyle='-',linewidth=1, color="b", label="Training Prediction", alpha=0.5)#Plot Predicted
                plt.legend(loc="upper left")
                #Bottom Plot-Testing
                plt.subplot(2,1,2)
                title = ("Testing("+information+"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.plot(test_label, linestyle='-',linewidth=2, color="g", label="Test Label", alpha=0.5)#Plot Acutal
                plt.plot(test_data, linestyle='-',linewidth=2, color="r", label="Testing Prediction", alpha=0.5)#Plot Predicted
                plt.legend(loc="upper left")
                #Commands
                plt.savefig(save_file)
                plt.close()
                print("Plot [c_combined_graph] Completed Successfully!")
                success = "Plot Completed Successfully!"
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("c_combined_graph ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return

#--------------------------------------------------------------------------#

    def plt_error_graph(self,filename, error_data, epochs, nn_type):
        '''
        Plot Measurement graph of Mean Square Error over Number of Total Epochs to runs sets and append to local storage

        error_graph(self,filename, error_data, epochs)
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type, 'error')

        #Fill array for prediction
        n_epochs=[]
        for i in range(epochs):
            n_epochs.append(i)

        while True:
            try:#Try/Catch for error Running
                print("Running [plt_error_graph()]!")
                title = ("Mean Squared Error("+information+"):["+ filename + "]")
                plt.xlabel("Number of Epochs")
                plt.ylabel("Error")
                plt.title(title, color="purple", fontsize=14)
                plt.plot(error_data, linestyle='-',linewidth=2, color="purple", label="MSE")#Plot Predicted
                #Global Cost Minimum
                #ymax = min(error_data)
                #xpos = error_data.index(ymax)
                #xmax = n_epochs[xpos]
                #full_data = ("Global Cost Minimum:[Value(" + ymax + "), Epoch Number(" + xmax + ")]")
                #plt.annotate(full_data, xy=(200, ymax), xytext=(200, ymax+5),arrowprops=dict(facecolor='black', shrink=0.05),)
                plt.legend(loc="upper left")
                plt.savefig(save_file)
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("plt_error_graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return

    def plt_bi_error_graph(self, filename, fw_error, bw_error, epochs, nn_type):
        '''
        Plots the error for bi directional networks in format;

        plt_bi_error_graph(self, filename, fw_error, bw_error, epochs, nn_type)
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type, 'error')

        #Fill array for prediction
        n_epochs=[]
        for i in range(epochs):
            n_epochs.append(i)

        while True:
            try:#Try/Catch for error Running
                print("Running [plt_error_graph()]!")
                title = ("Mean Squared Error("+information+"):["+ filename + "]")
                plt.xlabel("Number of Epochs")
                plt.ylabel("Error")
                plt.title(title, color="purple", fontsize=14)
                plt.plot(fw_error, linestyle='-',linewidth=2, color="green", label="Forward MSE")#Plot Predicted
                plt.plot(bw_error, linestyle='-',linewidth=2, color="purple", label="Backward MSE")#Plot Predicted
                plt.legend(loc="upper left")
                plt.savefig(save_file)
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("plt_error_graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return

#--------------------------------------------------------------------------#

    def plt_nn_train(self,label_data, rnn_data, lstm_data):
        '''
        Builds concatinated graph of all models current enrolled in Honours Project

        plt_nn_train(self,label_data, rnn_data, lstm_data)
        '''
        while True:
            try:#Try/Catch for error Running
                print("Running [plt_nn_train()]!")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.title("Model Training Comparison", color="m", fontsize=15)
                plt.plot(label_data, linestyle='-',linewidth=2, color="r", label="Actual Data", alpha=0.5)#Plot Predicted
                plt.plot(rnn_data, linestyle='-',linewidth=2, color="b", label="Recurrent Neural Network", alpha=0.5)#Plot Predicted
                plt.plot(lstm_data, linestyle='-',linewidth=2, color="g", label="Long Short-Term Memory", alpha=0.5)#Plot Predicted
                plt.legend(loc="upper left")
                plt.savefig("Recurrent_Network/Graphs/nn_train.png")#Save to File
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Train_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return DONE

    def plt_nn_test(self,label_data, rnn_data, lstm_data):
        '''
        Builds concatinated graph of all models current enrolled in Honours Project
        '''
        while True:
            try:#Try/Catch for error Running
                print("Running [plt_nn_test()]!")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.title("Model Testing Comparison", color="m", fontsize=15)
                plt.plot(rnn_data, linestyle='-',linewidth=2, color="b", label="Recurrent Neural Network", alpha=0.5)#Plot Predicted
                plt.plot(lstm_data, linestyle='-',linewidth=2, color="g", label="Long Short-Term Memory", alpha=0.5)#Plot Predicted
                plt.plot(label_data, linestyle='-',linewidth=2, color="r", label="Actual Data", alpha=0.5)#Plot Predicted
                plt.legend(loc="upper left")
                plt.savefig("Recurrent_Network/Graphs/nn_test.png")#Save to File
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Train_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return DONE

    def plt_nn_error(self, rnn_error, lstm_error):
        '''
        Plot error comparision between models 

        plt_nn_error(self, rnn_error, lstm_error):
        '''

#--------------------------------------------------------------------------#

    def bi_train_graph(self, filename, fw_data, bw_data, nn_type):
        '''
        Plotting of Bi-Directional Networks to single graph plotters
        '''
        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type, 'single-train')

        while True:
            try:#Try/Catch for error Running
                print("Running [bi_train_graph()]!")
                title = ("Bi-Directional Training:("+ information +"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(£)")
                plt.title(title, color="blue", fontsize=14)
                plt.plot(fw_data, linestyle='-',linewidth=2, color="blue", label="Forward Directional Data", alpha=0.5)
                plt.plot(bw_data, linestyle='-',linewidth=2, color="red", label="Backward Directional Data", alpha=0.5)
                plt.legend(loc="upper left")
                plt.savefig(save_file)#Save to File
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Train_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return DONE

    def bi_test_graph(self, filename, fw_data, bw_data, nn_type):
        '''
        Plotting of Bi-Directional Networks to single graph plotters
        '''
        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type, 'single-test')

        while True:
            try:#Try/Catch for error Running
                print("Running [bi_test_graph()]!")
                title = ("Bi-Directional Testing:("+ information +"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(£)")
                plt.title(title, color="blue", fontsize=14)
                plt.plot(fw_data, linestyle='-',linewidth=2, color="blue", label="Forward Directional Data")
                plt.plot(bw_data, linestyle='-',linewidth=2, color="red", label="Backward Directional Data")
                plt.legend(loc="upper left")
                plt.savefig(save_file)#Save to File
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("bi_test_graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return DONE

    def bi_c_train_graph(self, filename, fw_data, bw_data, label_data, nn_type):
        '''
        Bi-Directional Network plotters to compare training data versus label data;

        bi_c_train_graph(self, filename, fw_data, bw_data, label_data, nn_type)
        '''
        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type, 'compare-train')

        while True:
            try:#Try/Catch for error Running
                print("Running [bi_c_train_graph()]!")
                title = ("Bi-Directional Compare Training:("+ information +"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(£)")
                plt.title(title, color="blue", fontsize=14)
                plt.plot(fw_data, linestyle='-',linewidth=2, color="blue", label="Forward Directional Data")
                plt.plot(bw_data, linestyle='-',linewidth=2, color="red", label="Backward Directional Data")
                plt.plot(label_data, linestyle='-',linewidth=2, color="green", label="Expected Data")
                plt.legend(loc="upper left")
                plt.savefig(save_file)#Save to File
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("bi_c_train_graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return DONE

    def bi_c_test_graph(self, filename, fw_data, bw_data, label_data, nn_type):
        '''
        Bi-Directional Network plotters to compare testing data versus label data
        '''
        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type, 'compare-test')

        while True:
            try:#Try/Catch for error Running
                print("Running [bi_c_test_graph()]!")
                title = ("Bi-Directional Compare Testing:("+ information +"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(£)")
                plt.title(title, color="blue", fontsize=14)
                plt.plot(fw_data, linestyle='-',linewidth=2, color="blue", label="Forward Directional Data")
                plt.plot(bw_data, linestyle='-',linewidth=2, color="red", label="Backward Directional Data")
                plt.plot(label_data, linestyle='-',linewidth=2, color="green", label="Expected Data")
                plt.legend(loc="upper left")
                plt.savefig(save_file)#Save to File
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("bi_c_test_graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return DONE

    def bi_combined_graph(self, filename, train_fw_data, train_bw_data, test_fw_data, test_bw_data, train_label, test_label, nn_type):
        '''
        Bi-Directional Network plotters to compare all data in network cycle

        bi_combined_graph(self, filename, train_fw_data, train_bw_data, test_fw_data, test_bw_data, train_label, test_label, nn_type)
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type, 'combined')

        while True:
            try:#Try/Catch for error Running
                print("Begining [bi_combined_graph] Plotting")
                #Top Plot-Training
                plt.subplot(2,1,1)
                title = ("Training("+information+"):["+ filename + "]")
                plt.plot(train_label, linestyle='-',linewidth=1, color="g", label="Train Label", alpha=0.5)#Plot Acutal
                plt.plot(train_fw_data, linestyle='-',linewidth=1, color="b", label="Forward Prediction", alpha=0.5)#Plot Predicted
                plt.plot(train_bw_data, linestyle='-',linewidth=1, color="r", label="Backward Prediction", alpha=0.5)#Plot Predicted
                plt.legend(loc="upper left")
                #Bottom Plot-Testing
                plt.subplot(2,1,2)
                title = ("Testing("+information+"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.plot(test_label, linestyle='-',linewidth=2, color="g", label="Test Label", alpha=0.5)#Plot Acutal
                plt.plot(test_fw_data, linestyle='-',linewidth=1, color="b", label="Forward Prediction", alpha=0.5)#Plot Predicted
                plt.plot(test_bw_data, linestyle='-',linewidth=1, color="r", label="Backward Prediction", alpha=0.5)#Plot Predicted
                plt.legend(loc="upper left")
                #Commands
                plt.savefig(save_file)
                plt.close()
                print("Plot [bi_combined_graph] Completed Successfully!")
                success = "Plot Completed Successfully!"
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("bi_combined_graph ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return