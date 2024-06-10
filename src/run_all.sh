PROMPTS=(
    ""
    "no_sys"
    "no_sys_debias"
    "sys"
    "sys_debias"
)

for prompt in "${PROMPTS[@]}"; do
    echo $prompt
    python cal_bias.py -o T1 -r T2 -cppl -p "$prompt"
done