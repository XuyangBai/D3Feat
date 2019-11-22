for detector in "random" "sift" "harris" "iss"
do
    for i in 4 8 16 32 64 128 256 512
    do
        echo $detector $i
        python gen_3dmatch_keypts_baseline.py $detector $i
    done
done