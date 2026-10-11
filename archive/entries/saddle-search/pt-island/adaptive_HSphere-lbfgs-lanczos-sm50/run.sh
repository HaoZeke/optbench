#!/bin/env sh
################
################ IF NECESSARY, PLEASE EXPORT YOUR LIBRARIES HERE. 
################ 1) REMEMBER YOU NEED TO HAVE INSTALLED BLITZ++, BOOST, GSL AND MATLAB
################
################ 2) MAKE SURE THAT THE COMMAND matlab IS KNOWN BY YOUR SYSTEM, OTHERWISE
################ EDIT THIS BASH SCRIPT WITH THE FULL DIRECTION TO THE EXECUTABLE OF matlab
################ FOR EXAMPLE, A PARTICULAR CASE FOR matlab IS: 
################ ~/software/MATLAB2016b/bin/matlab
################
################ 3) VERIFY libstdc++.so.6 EXISTS: The bashscript must localize it and use it
################
################ 4) IF IT DOES EXITS, YOU CAN NOW RUN. 
################
################ 5) VERIFY THAT THIS SCRIPT IS EXECUTABLE, ORTHERWISE JUST DO
################ chmod 777 run.sh
################ 
################ 6) DOES YOUR COMPUTER ALLOWS hyperthreading?? THERE IS A LINE THAT TAKES CARE OF THAT, MARK IT AS true if so
################ OTHERWISE MARK IT AS false
################
################ 7) DO YOU WANT TO RUN A DEBUG?
################



#IF YOU NEED TO EXPORT YOUR LIBRARIES DO IT NOW.








#set to true if you want to use hyperthreading where its supported
#this will make the script count the number of logical processors instead
#of physical ones. Set to false if you want to count physical processors
HYPERTHREAD=false

#set this varible to the matlab path not needed if matlab command exists
#example MATLAB=/home/user/matlab/bin/matlab
MATLAB=

#This is for debugging purposes, just comment out if you want to run
#the script fully.
RUNDEBUG=false

##### DO YOU WANT TO RUN A FASTE TEST?
RUNFASTTEST=false

##### DOES YOUR SYSTEM RECOGNIZES THE COMMAND: ldconfig??
RECOGNISE=false

#################################################
#################################################
#################################################
##### DO YOU HAVE 25 FREE PROCESSORS NOW TO RUN?

YES=false

#################################################
#################################################
#################################################

#this checks if the matlab command exists if MATLAB variable is not set.
#if MATLAB variable is empty and matlab command not found we exit
if [ -z "$MATLAB" ];
then
 if command -v matlab >/dev/null 2>&1 ; then
     echo "matlab command found"
     MATLAB=matlab
 else
     echo "matlab not found, please set MATLAB variable. Aborting";
     exit 1
 fi
fi


#this should find the location of 64-bit version of libstdc++.so.6
if $RECOGNISE;
then
libstdcpath=$(ldconfig -p | grep  "^[[:space:]]*libstdc++.so.6.*x86-64" | cut -d\> -f2 | sed 's/ //g')
if [ ! -z "$libstdcpath" ] ;
then
    echo "found libstdc++.so.6 at $libstdcpath"
else
    echo "couldnt find libstdc++, aborting"
    exit 1
fi
else
## YOU HAVE TO WRITE THE ADDRESS TO libstdc++.so.6 IN HERE:
## DO IT EXACTLY AS: libstdcpath=/your/path/to/libstdc++.so.6
#libstdcpath=/your/path/to/libstdc++.so.6
libstdcpath=/usr/lib64/libstdc++.so.6
libstdcpath=/usr/local/MATLAB/R2017b/sys/os/glnxa64/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/local/MATLAB/R2017b/sys/os/glnxa64:$LD_LIBRARY_PATH
echo "You set libstdc++.so.6 at $libstdcpath"
fi







if $HYPERTHREAD ;
then
#this one counts the number of locical cpus, for example if it has hyperthreading
#and you want to use that then use this line.
numprocchecking=$(lscpu -p | grep -v "^#" | wc -l)
echo "using hyperthreading, number of processors is $numprocchecking"
else
#this one counts correctly the number of physical cpus even accross sockets
numprocchecking=$(lscpu -p | grep -v "^#" | sort -u -t, -k 2,4 |  wc -l)
echo "no hyperthreading, number of processors is $numprocchecking"
fi


if $YES ;
then
numproc=25
maxattempts=220
maxrealcor=25
else
if [ "$numprocchecking" -ge 20 ] && [ "$numprocchecking" -lt 24 ] ; then
numproc=20
maxattempts=275
maxrealcor=20
fi
if [ "$numprocchecking" -ge 10 ] && [ "$numprocchecking" -lt 19 ] ; then
numproc=10
maxattempts=550
maxrealcor=10
fi
if [ "$numprocchecking" -ge 5 ] && [ "$numprocchecking" -lt 9 ] ; then
numproc=5
maxattempts=1100
maxrealcor=5
fi
if [ "$numprocchecking" == 4 ] ; then
numproc=4
maxattempts=1375
maxrealcor=4
fi
if [ "$numprocchecking" == 3 ] ; then
numproc=2
maxattempts=2750
maxrealcor=2
fi
if [ "$numprocchecking" == 2 ] ; then
numproc=2
maxattempts=2750
maxrealcor=2
fi
fi


if $RUNFASTTEST;
then
numproc=2
maxattempts=1
maxrealcor=2
echo "THIS WILL RUN A FAST TEST WITH 2 PROCESSORS AND 1 MAX ATTEMP"
fi



echo "The number of processors to be used in this search is: " $numproc
echo "The (maximum) number of attempts is: "$maxattempts
echo "The number of available cores is: "$maxrealcor
product=$(echo "$(($numproc * $maxattempts))")
echo "The number of displacements on the hypersphere will be: " $product


########################### HERE FOR DEBUGGING PURPOSES ###############
#######################################################################
# this goes through if RUNDEBUG variable exist and is not empty string.

if $RUNDEBUG;
then
	blitzcheck=$(locate bzconfig.h)
        boostcheck=$(locate libboost)
	gslcheck=$(locate libgsl)
        if [ -z "$blitzcheck" ]; then echo "Looks that it is not possible to fin blitz"; else echo "Looks capable of finding blitz"; fi
        if [ -z "$boostcheck" ]; then echo "Looks that it is not possible to fin boost"; else echo "Looks capable of finding boost"; fi
        if [ -z "$blitzcheck" ]; then echo "Looks that it is not possible to fin GSL"; else echo "Looks capable of finding GSL"; fi
    echo "RUNDEBUG is set, aborting"
    exit
fi


tar xvfz SoftSaddle.tar.gz >/dev/null 2>&1
cd SoftSaddle/source_code/
address=$(pwd)
sed -i "s|~/SoftSaddle/disp_on_hypersphere_database|$address/data/disp_on_hypersphere_database/|g" SoftSaddle.m
#sed -i "s|/dev/shm/|$address/data/dev/shm/|g" SoftSaddle.m
#sed -ie "s|\/SoftSaddle_results\/|\/SoftSaddle_results\/$(echo $addresstotal | sed -r 's/./ /g')|g" SoftSaddle.m
#sed -ie "s|\/scratch\/|\/scratch\/$(echo $addresstotal | sed -r 's/./ /g')|g" SoftSaddle.m
#exit
cd data/disp_on_hypersphere_database
array=($(ls | sed 's/_.*//'))
for i in "${array[@]}"; do if [ "$i" -eq "$product" ]; then testing=true; break; else testing=false; fi; done;
cd ../../..
mkdir RESULTS
cd RESULTS
cp -r ../test_run/paper-result-reproduce/* .
cd adaptive_HSphere-lbfgs-davidson-sm50
echo "addpath('"$address"');" > startup.m
address2=$(echo "lib_path='"$address"';")
sed -i "s/lib_name=.*/lib_name='libdavidson';/g" iniConfig.m
sed -i "s|read_from_database = .*|read_from_database = $testing; % if true the points evenly distributed on the hypersherical surface are retreived from a database|g" iniConfig.m
sed -i "s|lib_path=.*|$address2|g" iniConfig.m
sed -i "s|parall_search_per_min=.*|parall_search_per_min=$numproc; % Number of simultaneous SP-searches in a batch|g" iniConfig.m
sed -i "s|max_attempts =.*|max_attempts = $maxattempts; % number of batches|g" iniConfig.m
sed -i "s|real_n_cores=.*|real_n_cores=$maxrealcor; % Number of computer cores dedicated to the parallel searches (real_n_cores >= parall_search_per_min)|g" iniConfig.m
cd ../adaptive_HSphere-lbfgs-lanczos-sm0
echo "addpath('"$address"');" > startup.m
address2=$(echo "lib_path='"$address"';")
sed -i "s|read_from_database = .*|read_from_database = $testing; % if true the points evenly distributed on the hypersherical surface are retreived from a database|g" iniConfig.m
sed -i "s|lib_path=.*|$address2|g" iniConfig.m
sed -i "s/lib_name=.*/lib_name='liblanczos';/g" iniConfig.m
sed -i "s|parall_search_per_min=.*|parall_search_per_min=$numproc; % Number of simultaneous SP-searches in a batch|g" iniConfig.m
sed -i "s|max_attempts =.*|max_attempts = $maxattempts; % number of batches|g" iniConfig.m
sed -i "s|real_n_cores=.*|real_n_cores=$maxrealcor; % Number of computer cores dedicated to the parallel searches (real_n_cores >= parall_search_per_min)|g" iniConfig.m
cd ../adaptive_HSphere-lbfgs-lanczos-sm50
echo "addpath('"$address"');" > startup.m
address2=$(echo "lib_path='"$address"';")
sed -i "s|read_from_database = .*|read_from_database = $testing; % if true the points evenly distributed on the hypersherical surface are retreived from a database|g" iniConfig.m
sed -i "s|lib_path=.*|$address2|g" iniConfig.m
sed -i "s/lib_name=.*/lib_name='liblanczos';/g" iniConfig.m
sed -i "s|parall_search_per_min=.*|parall_search_per_min=$numproc; % Number of simultaneous SP-searches in a batch|g" iniConfig.m
sed -i "s|max_attempts =.*|max_attempts = $maxattempts; % number of batches|g" iniConfig.m
sed -i "s|real_n_cores=.*|real_n_cores=$maxrealcor; % Number of computer cores dedicated to the parallel searches (real_n_cores >= parall_search_per_min)|g" iniConfig.m
cd ../..
echo "PREPARING THE LIBRARIES: COMPILE AND COPY THEM IN THE RIGHT FOLDERS"
cd libraries/davidsson_lib/to_compile/
g++ -O3 -shared -Wl,-soname,libdavidson.so  -o libdavidson.so -fPIC  -I/usr/lib64/blitz/include  *.cpp tridiag.c tridiagv.c  -lgsl -lgslcblas
cp libdavidson.so ../../../source_code/.
cd ../../libmorse/
g++ -O3 -shared -Wl,-soname,libmorse.so  -o libmorse.so -fPIC  -I/usr/lib64/blitz/include  *.cpp  -lgsl -lgslcblas
cp libmorse.so ../../source_code/.
cd ../lanczos_lib/to_compile/
g++ -O3 -shared -Wl,-soname,liblanczos.so  -o liblanczos.so -fPIC  -I/usr/lib64/blitz/include  *.cpp tridiag.c tridiagv.c  -lgsl -lgslcblas
cp liblanczos.so ../../../source_code/.

cd ../../../RESULTS/


cp ../test_run/get_data_for_web_benchmark.m adaptive_HSphere-lbfgs-davidson-sm50/.
cp ../test_run/get_data_for_web_benchmark.m adaptive_HSphere-lbfgs-lanczos-sm0/.
cp ../test_run/get_data_for_web_benchmark.m adaptive_HSphere-lbfgs-lanczos-sm50/.
echo "STARTS THE COMPUTATION PROCESS; THIS COULD TAKE A LONG TIME"
cd adaptive_HSphere-lbfgs-davidson-sm50/.
pwd
LD_PRELOAD="$libstdcpath"  $MATLAB -r -nosplash -nodesktop "SoftSaddle" #> adaptive_HSphere-lbfgs-davidson-sm50.out 2> adaptive_HSphere-lbfgs-davidson-sm50.err
address4=$(ls SoftSaddle_results/morse/)
sed -i "s|my_path2    =.*|my_path2    ='SoftSaddle_results/morse/$address4';|g" get_data_for_web_benchmark.m
LD_PRELOAD="$libstdcpath"  $MATLAB -r -nosplash -nodesktop "get_data_for_web_benchmark";
rm -rf dev
cd ../adaptive_HSphere-lbfgs-lanczos-sm0/.
pwd
LD_PRELOAD="$libstdcpath"  $MATLAB -r -nosplash -nodesktop "SoftSaddle"  #> adaptive_HSphere-lbfgs-lanczos-sm0.out 2> adaptive_HSphere-lbfgs-lanczos-sm0.err
address4=$(ls SoftSaddle_results/morse/)
sed -i "s|my_path2    =.*|my_path2    ='SoftSaddle_results/morse/$address4';|g" get_data_for_web_benchmark.m
LD_PRELOAD="$libstdcpath"  $MATLAB -r -nosplash -nodesktop "get_data_for_web_benchmark";
rm -rf dev
cd ../adaptive_HSphere-lbfgs-lanczos-sm50/.
pwd
LD_PRELOAD="$libstdcpath"  $MATLAB -r -nosplash -nodesktop "SoftSaddle" #> adaptive_HSphere-lbfgs-lanczos-sm50.out 2> adaptive_HSphere-lbfgs-lanczos-sm50.err 
address4=$(ls SoftSaddle_results/morse/)
sed -i "s|my_path2    =.*|my_path2    ='SoftSaddle_results/morse/$address4';|g" get_data_for_web_benchmark.m
LD_PRELOAD="$libstdcpath"  $MATLAB -r -nosplash -nodesktop "get_data_for_web_benchmark";
rm -rf dev
echo "ALL computation finished. Check in SoftSaddle/RESULTS folder."

#BASHSCRIPT WRITTEN BY CARLOS ARGAEZ (carlos@hi.is). SHOULD YOU FIND ANY PROBLEM OR QUESTION ON THE WAY, PLEASE USE carlos@hi.is TO REQUEST HELP.

