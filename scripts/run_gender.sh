# MODEL="taide/TAIDE-LX-7B-Chat"
# MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL="google/gemma-2-9b-it"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo Model: "$MODEL"
echo Combination: "T1-T2"
python src/cal_bias.py  -cppl \
                        -m $MODEL \
                        -tp "data/gender/target_gender.csv" \
                        -o T1 \
                        -r T2 \
                        -bp "data/gender/label_data_male.csv" \
                        -ap "data/gender/male-Attribute.csv" \
                        -rd "result/gender"

echo Model: "$MODEL"
echo Combination: "T2-T1"
python src/cal_bias.py  -cppl \
                        -m $MODEL \
                        -tp "data/gender/target_gender.csv" \
                        -o T2 \
                        -r T1 \
                        -bp "data/gender/label_data_female.csv" \
                        -ap "data/gender/female-Attribute.csv" \
                        -rd "result/gender"