# MODEL="taide/TAIDE-LX-7B-Chat"
# MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL="google/gemma-2-9b-it"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"

NUM_GPU=4

# Declare an associative array
declare -A origin_dict

# Add key-value pairs to the dictionary
origin_dict=(
    [T1]="B"
    [T2]="W"
    [T4]="NT"
    [T5]="hakka"
)

replace_array=(
    T1
    T2
    T3
    T4
    T5
)

gpu_index=0
for origin in "${!origin_dict[@]}"; do
    for replace in "${replace_array[@]}"; do
        # Skip this loop if origin == replace
        if [[ "$origin" == "$replace" ]]; then
            # Continue to the next iteration of the loop
            continue
        fi
        echo Model: "$MODEL"
        echo Combination: "$origin"-"$replace"
        CUDA_VISIBLE_DEVICES=$gpu_index python src/cal_bias_single_gpu.py -cppl \
                            -m $MODEL \
                            -tp "data/ethnicity/target_ethnicity.csv" \
                            -o $origin \
                            -bp "data/ethnicity/label_data_${origin_dict[$origin]}.csv" \
                            -ap "data/ethnicity/${origin_dict[$origin]}-Attribute.csv" \
                            -r $replace \
                            -rd "result/ethnicity" \
                            -comb &

        gpu_index=$(( (gpu_index + 1) % $((NUM_GPU)) ))
    done
    wait
done