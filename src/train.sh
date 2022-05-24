range=(1 2 3 4 5)
for i in "${range[@]}"
do
    batch_size=4
    len=`cat ../data/dataset_5fold/fold_$i/train.csv | wc -l`
    steps=$(($len/$batch_size/2)) # steps of 0.5 epoch
    `
    python run_glue.py \
        --do_train \
        --do_eval \
        --model_name_or_path=microsoft/deberta-base \
        --max_seq_length 512 \
        --per_device_train_batch_size $batch_size \
        --learning_rate 2e-5 \
        --warmup_ratio 0.0 \
        --num_train_epochs 5 \
        --output_dir ../models/5fold/deberta/fold-$i \
        --overwrite_output_dir \
        --train_file ../data/dataset_5fold/fold_$i/train.csv \
        --validation_file ../data/dataset_5fold/fold_$i/dev.csv \
        --save_strategy steps \
        --save_steps $steps \
        --logging_strategy steps \
        --logging_steps $steps \
        --evaluation_strategy steps \
        --eval_steps $steps \
        --save_total_limit=1 \
        --metric_for_best_model=macro-f1 \
        --load_best_model_at_end=True \
    `
done