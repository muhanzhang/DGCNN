#! /bin/bash
# This is a shell script automatically running 100 experiments on some dataset. The hyperparameters are those being used in the paper.
# Note that we chose to not use validation set to select the best net, but just let it run until reaching the maxEpoch and report the final test accuracy (by setting -valRatio 0). This is because graph datasets are usually too small for its validation split to be representative of its test split. We have tested that the validation splits are hardly useful here, thus giving them up.
# *author: Muhan Zhang, Washington University in St. Louis

data="${1}"
GPU=${2}
save="result"
fileName="${save}/${data}/testAcc"
common="-gpu ${GPU} -trainRatio 0.9 -valRatio 0"
rm ${save}/${data}/testAcc
rm ${save}/${data}/trainAcc
echo $common
echo ${fileName}
echo "A new run" >> $fileName
start=`date +%s`
for i in $(seq 1 10)  # to run all random cv splits in one time
do
  for j in $(seq 1 10)
  do
    echo "...............$i-$j..............."
    case ${data} in
    DD)
      th main.lua -dataName DD -maxNodeLabel 89 -learningRate 1e-5 -fixed_shuffle "${i}_${j}" -maxEpoch 200 -save "${save}" $common -outputChannels '32 32 32 1'

    ;;
    MUTAG)
      th main.lua  -maxNodeLabel 7 -learningRate 1e-4 -fixed_shuffle "${i}_${j}" -maxEpoch 300 -save "${save}" $common -outputChannels '32 32 32 1'

    ;;
    ptc)
      th main.lua -dataName ptc -maxNodeLabel 22 -learningRate 1e-4 -fixed_shuffle "${i}_${j}" -maxEpoch 200 -save "${save}" $common -outputChannels '32 32 32 1'

    ;;
    NCI1)
      th main.lua -dataName NCI1 -maxNodeLabel 37 -learningRate 1e-4 -fixed_shuffle "${i}_${j}" -maxEpoch 200 -save "${save}" $common -outputChannels '32 32 32 1'

    ;;
    proteins)
      th main.lua -dataName proteins -maxNodeLabel 3 -learningRate 1e-5 -fixed_shuffle "${i}_${j}" -maxEpoch 100 -save "${save}" $common -outputChannels '32 32 32 1'

    ;;
    COLLAB)
      th main.lua -dataName COLLAB -nClass 3 -learningRate 1e-4 -fixed_shuffle "${i}_${j}" -maxEpoch 300 -save "${save}" $common -outputChannels '32 32 32 1' -nodeLabel nDegree -k 130

    ;;
    IMDBBINARY)
      th main.lua -dataName IMDBBINARY -learningRate 1e-4 -fixed_shuffle "${i}_${j}" -maxEpoch 300 -save "${save}" $common -nodeLabel nDegree -k 31 -outputChannels '32 32 32 1'

    ;;
    IMDBMULTI)
      th main.lua -dataName IMDBMULTI -nClass 3 -learningRate 1e-4 -fixed_shuffle "${i}_${j}" -maxEpoch 500 -save "${save}" $common -outputChannels '32 32 32 1' -nodeLabel nDegree -k 22

    ;;
    *) echo 'Dataset does not exist.'
    ;;
    esac
  done
done
stop=`date +%s`
echo "End of this run" >> $fileName
echo "The total running time is $[stop - start] seconds."
echo "The accuracy results for ${data} are as follows:"
cat ${fileName}
