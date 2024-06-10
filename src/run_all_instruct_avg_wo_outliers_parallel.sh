MODELS=(
    # "yentinglin/Tawan-LLM-7B-v2.1-chat" 
    "yentinglin/Taiwan-LLM-7B-v2.0.1-chat"
    # "yentinglin/Taiwan-LLM-13B-v2.0-chat"
    "MediaTek-Research/Breeze-7B-Instruct-v0_1"
    # "/work/u3659277/s3-local/models/llama2-7b-ccw_cp-tw9_tv-0.5tulu2-0.5xwin"
    # "/work/u3659277/s3-local/models/llama2-13b-ccw_cp-tw9_tv-0.5tulu2-0.5xwin"
    # "/work/u3659277/s3-local/models/llama2-7b-ccw_cp-n0_tv_0.5xwin-0.5tulu"
    # "/work/u3659277/s3-local/models/llama2-13b-ccw_cp-n0_tv_0.5xwin-0.5tulu"
    # "/work/u3659277/s3-local/models/llama2-7b-ccw_cp-n1_tv_0.5xwin-0.5tulu"
    # "/work/u3659277/s3-local/models/llama2-13b-ccw_cp-n1_tv_0.5xwin-0.5tulu"
    # "/work/u3659277/s3-local/models/llama2-7b-ccw_cp-p1_tv_ft-b8.3-capybara-mixed-e3_spin-1"
)

PROMPTS=(
    "0"
    "1"
    "2"
    "3"
    "4"
    "5"
    "6"
    "7"
    "8"
    "9"
    "10"
    "11"
)

TP="/home/u3659277/master-thesis/TWBias/data/ethical_group/target_ethical_group.csv"
BP="/home/u3659277/master-thesis/TWBias/data/ethical_group/label_data_B.csv"
AP="/home/u3659277/master-thesis/TWBias/data/ethical_group/B-Attribute.csv"

# 不用算PPL的時候的BP
# modelname=$(basename "$MODELS")
# echo "Extracted filename: $modelname"
# -bp "/home/u3659277/master-thesis/TWBias/data/ethical_group/result_wo_outliers/${modelname}/
# ${origin_dict[$origin]}_$replace/${modelname}_${origin_dict[$origin]}_${replace}_output_$prompt.csv"


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

log_file="/home/u3659277/master-thesis/TWBias/data/ethical_group/result_wo_outliers/time_taide.txt"

for model in "${MODELS[@]}"; do
    for origin in "${!origin_dict[@]}"; do
        for replace in "${replace_array[@]}"; do
            # Skip this loop if origin == replace
            if [[ "$origin" == "$replace" ]]; then
                # Continue to the next iteration of the loop
                continue
            fi
            total_prompts=${#PROMPTS[@]}
            concurrent_jobs=6

            # Get the start time
            # start_time=$(date +%s)
            for ((i = 0; i < total_prompts; i += concurrent_jobs)); do
                for ((j = i; j < i + concurrent_jobs; j++)); do
                    prompt=${PROMPTS[$j]}
                    echo Prompt: "$prompt"
                    echo Model: "$model"
                    echo Combination: "$origin"-"$replace"
                    python cal_bias_auto_avg.py -cppl \
                                        -m $model \
                                        -p $prompt \
                                        -tp $TP \
                                        -o $origin \
                                        -bp "/home/u3659277/master-thesis/TWBias/data/ethical_group/label_data_${origin_dict[$origin]}.csv" \
                                        -ap "/home/u3659277/master-thesis/TWBias/data/ethical_group/${origin_dict[$origin]}-Attribute.csv" \
                                        -r $replace \
                                        -comb 
                done
                # wait for all background jobs to complete
                wait
            done

            # Get the end time
            end_time=$(date +%s)
            # Calcuate the duration
            duration=$((end_time - start_time))
            # Log the command and its duration to the log file
            echo "Model: $model" >> "$log_file"
            echo "Combination: $origin-$replace" >> "$log_file"
            echo "Duration: $duration seconds" >> "$log_file"
            echo "---------------------------" >> "$log_file"
        done
    done
done