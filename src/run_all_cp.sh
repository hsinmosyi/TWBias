MODELS=(
    "/work/u3659277/s3-local/models/llama2-7b-ccw-n0_e0-s28044"
    "/work/u3659277/s3-local/models/llama2-13b-ccw-n0_e0-s28045"
    "/work/u3659277/s3-local/models/llama2-7b-ccw-n1_e0-s23932"
    "/work/u3659277/s3-local/models/llama2-13b-ccw-n1_e0-s23932"
)

TP="/home/u3659277/master-thesis/TWBias/data/gender/target_gender.csv"

ORIGIN=$1
REPLACE=$2
BP="/home/u3659277/master-thesis/TWBias/data/gender/label_data_male_final_240220.csv"
AP="/home/u3659277/master-thesis/TWBias/data/gender/Male-Attribute.csv"

for model in "${MODELS[@]}"; do
    echo Model: "$model"
    echo Combination: "$ORIGIN"-"$REPLACE"
    python cal_bias_cp.py  -cppl \
                        -m $model \
                        -tp $TP \
                        -o $ORIGIN \
                        -bp $BP \
                        -ap $AP \
                        -r $REPLACE

done