MODELS=(
    # "yentinglin/Taiwan-LLM-7B-v2.1-chat" 
    "yentinglin/Taiwan-LLM-7B-v2.0.1-chat"
    # "yentinglin/Taiwan-LLM-13B-v2.0-chat"
    # "MediaTek-Research/Breeze-7B-Instruct-v0_1"
    # "/work/u3659277/s3-local/models/llama2-7b-ccw_cp-tw9_tv-0.5tulu2-0.5xwin"
    # "/work/u3659277/s3-local/models/llama2-13b-ccw_cp-tw9_tv-0.5tulu2-0.5xwin"
    # "/work/u3659277/s3-local/models/llama2-7b-ccw_cp-n0_tv_0.5xwin-0.5tulu"
    # "/work/u3659277/s3-local/models/llama2-13b-ccw_cp-n0_tv_0.5xwin-0.5tulu"
    # "/work/u3659277/s3-local/models/llama2-7b-ccw_cp-n1_tv_0.5xwin-0.5tulu"
    # "/work/u3659277/s3-local/models/llama2-13b-ccw_cp-n1_tv_0.5xwin-0.5tulu"
    # "/work/u3659277/s3-local/models/llama2-7b-ccw_cp-p1_tv_ft-b8.3-capybara-mixed-e3_spin-1"
)

PROMPTS=(
    # ""
    "no_sys"
    # "no_sys_debias"
    # "sys"
    # "sys_debias"
)

TP="/home/u3659277/master-thesis/TWBias/data/ethical_group/ethical_group_target.csv"

ORIGIN=$1
REPLACE=$2
BP="/home/u3659277/master-thesis/TWBias/data/ethical_group/label_data_B_final_240223.csv"
AP="/home/u3659277/master-thesis/TWBias/data/ethical_group/B-Attribute.csv"

for model in "${MODELS[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        echo Prompt: "$prompt"
        echo Model: "$model"
        echo Combination: "$ORIGIN"-"$REPLACE"
        python cal_bias_auto_avg.py  -cppl \
                            -m $model \
                            -p "$prompt" \
                            -tp $TP \
                            -o $ORIGIN \
                            -bp $BP \
                            -ap $AP \
                            -r $REPLACE \
                            -comb
    done
done