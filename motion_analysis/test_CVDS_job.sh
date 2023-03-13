export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

for i in /path/to/folder/of/videos/to/analyze/*mp4; do
    # echo "$(basename "$i")" 
    /path/to/OpenFace/build/bin/FeatureExtraction -f "${i}" -pose -aus -2Dfp -3Dfp -pdmparams -out_dir /playpen-nas-ssd/akshay/UNC_Google_Physio/motion_analysis/cdvs_high_au -of "$(basename "$i")";
done
