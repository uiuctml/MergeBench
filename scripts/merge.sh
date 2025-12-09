# Model soup
python ./merging/main.py --algo TaskArithmetic --scaling-coef 0.2 --base-model meta-llama/Llama-3.2-3B

# Task arithmetic
python ./merging/main.py --algo TaskArithmetic --scaling-coef 0.4 --base-model meta-llama/Llama-3.2-3B

# Fisher Merging
ALL_TASKS=DartMath-WildguardMix-MagiCoder-Aya-Tulu3IF

for TASK in DartMath WildguardMix MagiCoder Aya Tulu3IF; do
  deepspeed --master_port=61001 --include=localhost:0,1,2,3 ./merging/main.py \
    --algo Fisher \
    --base-model meta-llama/Llama-3.2-3B \
    --task_names $TASK \
    --save_group $ALL_TASKS \
    --fisher_only \
    --model_coeff 1
done

python ./merging/main.py \
    --algo Fisher \
    --base-model meta-llama/Llama-3.2-3B \
    --task_names $ALL_TASKS \
    --save_group $ALL_TASKS \
    --merge_only \
    --keep_checkpoints \
    --model_coeff 1

# RegMean
python ./merging/main.py --algo RegMean --base-model meta-llama/Llama-3.2-3B --task_names DartMath-WildguardMix-MagiCoder-Aya-Tulu3IF --reduction 0.5

# RegMeanPlusPlus
python ./merging/main.py --algo RegMeanPlusPlus --base-model meta-llama/Llama-3.2-3B --task_names DartMath-WildguardMix-MagiCoder-Aya-Tulu3IF --reduction 0.1

# TIES Merging
python ./merging/main.py --algo TIES --base-model meta-llama/Llama-3.2-3B --K 0.3 --scaling-coef 0.4

# DARE
python ./merging/main.py --algo RegMean --base-model meta-llama/Llama-3.2-3B --p 0.9 --scaling-coef 0.4

# Consensus TA
python ./merging/main.py --algo RegMean --base-model meta-llama/Llama-3.2-3B --scaling-coef 0.4

# Dataless Localize-and-Stitch
python ./merging/main.py --algo LocalizeAndStitch --base-model meta-llama/Llama-3.2-3B --sparsity 0.1 --dataless

# Localize-and-Stitch
python ./merging/main.py --algo LocalizeAndStitch --base-model meta-llama/Llama-3.2-3B  --lr 1e8 --sparsity 0.1 --n_epochs 1
