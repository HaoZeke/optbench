#!/usr/bin/env python
import os, subprocess
import shutil
import sys
from itertools import groupby # this is python standard library package
import numpy as np
import contextlib
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from pyamff.config import ConfigClass #

def edit_config(total_path,flag,value):
    # inputs are a file path (config.ini, a flag (e.g "hidden_layers"), and value we want the flag to have (i.e "5 2")
    # Will we request the user to always give us an initial set of hyperparameters?
    with open(total_path, "r") as f: # read config.ini in some /tanh/..._model/config.ini
        lines = f.readlines() # read that file - right now it is exactly same as config in path_initial
    change_line = 0 # keep track of whether we have changed the hyperparam flag line or not
    with open(total_path, "w") as f: #open the file again with write (i.e overwriting now)
        for line in lines: # for each line in the file 
            if line.startswith(flag): #if that line has our flag
                f.write(flag+ " = "+value+"\n") #overwrite the flag with our value
                change_line = 1 # we have updated the config.ini 
            else: # otherwise
                f.write(line) # rewrite the flag content as in original config.ini
        # the code below automatically adds to end of config.ini, which will error
        if change_line == 0: # have to resolve this requirement with yaml defaults somehow
            sys.exit('please set an initial '+flag+' in config.ini')
        f.close() #close the file
    return 0

# In a directory, need config.ini, fpParas, and traj file to run pyamff

### condsidering we don't want a matrix, and will change only one hyperparameter at a time;
### we can rely on just 1 list to name directories ...
def one_hyperparam_space_set_up(hp_list,flag='optimizer_type',num_dirs=None,traj_name = 'train.traj',sub_file = None,run_no_queue=False):
    # copy original user config, and only edit 1 line 
    for hp in range(len(hp_list)):
        dir_name = 'user-'+flag +'_' + hp_list[hp]
        model_dir =(path_initial+"/"+dir_name+"_model")
        #print ("model_dir: ",model_dir)
        if os.path.exists(model_dir)==False: #if the directory does not exist -i.e calc not set up yet
           os.mkdir(model_dir) #make the directory 
        os.chdir(model_dir) # go to the directory
        if num_dirs ==None: # if running set up models
            num_dirs = len(os.listdir()) #run each directory in /optimizer
        #print ("num_dirs: ",num_dirs)
        for dir in range(num_dirs):
            dir_path = model_dir+"/"+str(dir) # 0,1,2,3,...
            if os.path.exists(dir_path)==False: #if the directory does not exist -i.e calc not set up yet
               os.mkdir(dir_path) #make the directory 
            os.chdir(dir_path)
            #now we are in /Rprop/0, now we need to copy the config,traj,fpParas if set up; or submission file if running
            if sub_file and run_no_queue == False: # assuming SGE queue
                # run the directory via queue
                shutil.copyfile(path_initial+"/"+sub_file ,dir_path+"/"+sub_file) # copy the submission script
                run_command = "qsub "+sub_file
                os.system(run_command) # submit to queue
            elif run_no_queue == True: # will run directly
                # run the directory without a queue on the head node
                #for optbench
                run_command = 'pyamff'
                os.system(run_command) # run it
            else: # setup calc
                # copy the config, fpParas and traj file to the made directory
                shutil.copyfile(path_initial+"/config.ini" ,dir_path+"/config.ini") # copy the original config to unique model directory
                shutil.copyfile(path_initial+"/fpParas.dat" ,dir_path+"/fpParas.dat") # copy the orginal fpParas
                shutil.copyfile(path_initial+"/"+traj_name ,dir_path+"/"+traj_name) # copy the trajectory file 
                # Edit the config.ini with needed hyperparameters    
                edit_config("config.ini",flag,hp_list[hp]) #i.e in the new directory change optimizer        
    return 0

#Function to set up calculations for all the hyperparameters (lists will be empty if user did not optimize that hyperparam)
def setup_calc(opt_list,traj_name = 'train.traj',num_dirs=3):
    one_hyperparam_space_set_up(opt_list,flag='optimizer_type',traj_name=traj_name,num_dirs=num_dirs)
    return 0

##### Run all the models based on queue or directly running depending upon -rq or -r given
def run_models(opt_list,traj_name='train.traj',sub_file='run.sh',run_no_queue=False,num_dirs=None):#important to pass None
    #This function will only be called when the user gives a submission script
    #Don't set up any calcualtions, pass the sub file to func
    if sub_file and run_no_queue == False:
        one_hyperparam_space_set_up(opt_list,flag='optimizer_type',num_dirs=None,sub_file = sub_file)
    else: #sub_file and now run_no_queue = True
        one_hyperparam_space_set_up(opt_list,flag='optimizer_type',sub_file = sub_file,run_no_queue=run_no_queue)
    return 0
    
#MOST RUN FUNCTION
# given a path of trained model; 
def report_path_result(path):
    ###Modified for ml-opt-benchmark
    os.chdir(path) # go the directory
    if os.path.exists('./pyamff.log') ==True:
        lines = os.popen('tail -2 pyamff.log').read()
    else:
        no_log = "please check "+path+ ", it does not have pyamff.log"
        result = [path,-1.0,-1.0,-1.0,1]
        return result
    lines = lines.split('\n')
    training_data = lines[0].lstrip()
    training_data = training_data.split()
    try:
        epochs = int(training_data[0])
    except:
        print ("error path: ",path)
        return [path,-1.0,-1.0,-1.0,1]
    Training_U_rmse = training_data[4]
    Training_F_rmse = training_data[5]
    if lines[1].startswith('Max'):
        converge = 1 #it did not converge
    else:
        converge = 0 # it converged
    result = [path,epochs,float(Training_U_rmse),float(Training_F_rmse),converge] #store as list
    return result

# for each trained model, summarize results into a text file
# dictionary is Felix inspired
def write_all_results(main_path,mode='no_scan'):
    summary = open("summary_optimizers.txt", "w") # open the file
    list_of_all_data = []
    opt_wise_data ={}
    summary.write(" FileName        Epoch        Training Energy RMSE         Training Force RMSE        Converge\n")
    benchmark = open("benchmark.dat", "w")
    for file in os.listdir(): # for each file in main_path
        if file.endswith("_model"): # if the file ends in "_model"
           opt_file = main_path+"/"+file
           os.chdir(opt_file)
           key = os.path.relpath(opt_file,start=main_path).replace("_model",'') # opt_file but less messy than full path
           opt_wise_data[key] =[[],[],[],[],[]] # each opt as key
           for num_dir in os.listdir():
                os.chdir(num_dir)
                path = os.path.join(opt_file,num_dir) #get_path to run directory
                rel_path = os.path.relpath(path,start = main_path)
                temp_path_result = report_path_result(path)
                for i in range (5): # for each i.e Filename,Train_U_rmse,Train_F_rmse,converge
                    if i ==4:
                        summary.write("{:.6f}".format(temp_path_result[i])+"       ") # call report_path_result and write the results
                        if temp_path_result[i] ==1: #if did not converge
                           opt_wise_data[key][i].append(temp_path_result[i])
                    elif i>0 : # if not path (i.e energy or force RMSE)
                        summary.write("{:.6f}".format(temp_path_result[i])+"       ") # call report_path_result and write the results
                        #file_data.append(report_path_result(path)[i])
                        opt_wise_data[key][i].append(temp_path_result[i])
                    else: #i =0, filename
                        summary.write(rel_path+"      ")
                        opt_wise_data[key][i].append(rel_path)
                        # for the filename, store results (i.e E, F rmse);
                        file_data = []
                        file_data.append(rel_path)
                summary.write("\n")  # after written 1 line, start new line
                os.chdir(opt_file)
    summary.write("\nOptimizer Wise Results: Avg_Epoch  Avg_Final_U_RMSE   Avg_Final_F_RMSE\n")
    summary.write("-------------------------------------- \n")
    for key in opt_wise_data:
        avg_epoch = np.mean(opt_wise_data[key][1])
        avg_train_U_rmse = np.mean(opt_wise_data[key][2])
        avg_train_F_rmse = np.mean(opt_wise_data[key][3])
        benchmark.write('mean_epoch '+str(avg_epoch)+'\n')
        benchmark.write('median_epoch '+str(np.median(opt_wise_data[key][1])) +'\n')
        benchmark.write('mean_energy_rmse '+str(avg_train_U_rmse)+'\n')
        benchmark.write('median_energy_rmse '+str(np.median(opt_wise_data[key][2]))+'\n')
        benchmark.write('mean_force_rmse -1.0\n')
        benchmark.write('median_force_rmse -1.0\n')
        benchmark.write('nfailed '+str(len(opt_wise_data[key][4]))+" \n")
        opt_result_to_log = key+": "+str(avg_epoch)+", "+ str(avg_train_U_rmse)+ ", "+str(avg_train_F_rmse)+" \n"
        summary.write(opt_result_to_log)
        summary.write("Times optimizer reached max epoch: "+str(len(opt_wise_data[key][4]))+" \n")
        summary.write("-------------------------------------- \n")
    benchmark.close()
    return 0

def get_log_plot_data(file):
    temp = open(file,'r')
    get_train_line = 'tail -2 '+file+" | head -1"
    last_train_line = os.popen(get_train_line).read()
    split_training_line = last_train_line.split()
    num_epoch_plot = int(split_training_line[0])+1
    num_head = num_epoch_plot-1
    to_plot_get_data = "tail -" +str(num_epoch_plot)+ " "+file+" "+"| head -"+str(num_head)
    raw_training_data = os.popen(to_plot_get_data).read()
    training_data = raw_training_data.split()
    epochs_arr = (list(map(int,training_data[0::6]))) #most 'pythonic' and fast method apparently
    u_rmse_arr = (list(map(float,training_data[4::6]))) #most 'pythonic' and fast method apparently
    f_rmse_arr = (list(map(float,training_data[5::6])))
    return epochs_arr,u_rmse_arr,f_rmse_arr

def plot_log(epoch,u_rmse,f_rmse):
    #plot the Energy RMSEs
    fig1,ax1 = plt.subplots()
    plt.plot(epoch,u_rmse,label='energy rmse',color='b')
    plt.scatter(epoch,u_rmse)
    plt.legend()
    plt.title('Energy RMSE vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Energy RMSE')
    fig1.savefig('energy_epoch.png')
    #Now the Force RMSEs
    fig2,ax2 = plt.subplots()
    plt.plot(epoch,f_rmse,label='force rmse',color='green')
    plt.scatter(epoch,f_rmse,color='green')
    plt.legend()
    plt.title('Force RMSE vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Force RMSE')
    fig2.savefig('force_epoch.png')
    return 0


def main():
    run_only = False # will be used only when -rq is given
    run_only_no_queue = False #True when -r is given
    num_dirs = None
    if len(sys.argv) > 1: # i.e more flags than just "python hyperparam_opt.py"
        if sys.argv[1] == "-s": # this 1 is array index; i.e python hyperparam_opt.py -s has len(sys.argv) = 2
           write_all_results(path_initial)
           return 0
        if sys.argv[1].isdigit():
            num_dirs = int(sys.argv[1])
        if sys.argv[1] =="-rq":  # if the user gives a -r 
            #sample usage: "python hyperparam_opt.py -r a.sub"
            print ("Note: we assume an SGE queue. ") #Have to make this for all common clusters
            run_only = True 
            run_file_name = sys.argv[2] # get the submission file name
            if run_file_name not in os.listdir():
                print ("the given run file is not in this directoecty")
                return 1
        if sys.argv[1] =="-r":  # if the user gives a -r 
            run_only_no_queue = True 
    traj_name = config.config['trajectory_file']# get the traj filename from config.ini
    opt_list = ['ADAM']
    if run_only == True: # run pre-set up calculations
        print ("submitting set up models to train")
        run_models(opt_list, sub_file=run_file_name,num_dirs=None)    
    if run_only_no_queue == True: # run pre-set up calculations
        print ("starting to run set up models to train")
        run_models(opt_list, sub_file=None,run_no_queue=True,num_dirs=None)    
    if run_only ==False and run_only_no_queue ==False: #only set up calculations
       print ("setting up the ml models")
       if num_dirs ==None:
          num_dirs =3
       setup_calc(opt_list,traj_name=traj_name,num_dirs=num_dirs)
    print ('done')
    os.chdir(path_initial)  # for sanity check, return python to main dir
    return 0 

if __name__ == "__main__":
    ### Note: path_inital is defined globally - the directory from which we are running 
    ### We also assume the user has config.ini,fpParas,traj files in the directory from which we are running
    path_initial = os.getcwd() 
    if sys.argv[1] == '-p':
            file = sys.argv[2]
            epochs_arr,u_rmse_arr,f_rmse_arr = get_log_plot_data(file)
            plot_log(epochs_arr,u_rmse_arr,f_rmse_arr)
            print ('logfile plots made')
    else:
       with contextlib.redirect_stdout(None):
           config = ConfigClass()
           config.initialize()
       os.remove('pyamff.log') #remove init pyamff.log, not useful for this
       main()
