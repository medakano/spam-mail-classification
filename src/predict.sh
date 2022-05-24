range=(1 2 3 4 5)
models=("deberta" "roberta-base" "xlnet-base")
for model in "${models[@]}"
    do
    for i in "${range[@]}"
        do
            `
            python run_glue_logits.py \
                --do_predict \
                --model_name_or_path=../models/5fold/$model/fold-$i \
                --max_seq_length 512 \
                --output_dir ../predictions/5fold/${model}_${i}/ \
                --train_file ../data/dataset//test.csv \
                --validation_file ../data/dataset/test.csv \
                --test_file ../data/dataset/test.csv \
            `
        done
    done