for SEED in 42;
do python train_retacred.py --model_name_or_path bert-base-cased --input_format typed_entity_marker_punct --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name tacred-bert-base --save_dir saved_models_ber_base;
done;

for SEED in 42;
do python train_retacred.py --model_name_or_path bert-base-cased --input_format typed_entity_marker_pos_punct --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name tacred-bert-base-pos --save_dir saved_models_bert_base_pos;
done;

# for SEED in 78 23 61;
# do python train_tacred.py --model_name_or_path bert-large-cased --input_format typed_entity_marker_punct --seed $SEED --run_name tacred-bert-large;
# done;

# for SEED in 78 23 61;
# do python train_tacred.py --model_name_or_path bert-large-cased --input_format typed_entity_marker_pos_punct --seed $SEED --run_name tacred-bert-large-pos;
# done;

# for SEED in 78 23 61;
# do python train_retacred.py --model_name_or_path bert-large-cased --input_format typed_entity_marker_punct --seed $SEED --run_name retacred-bert-large;
# done;

# for SEED in 78 23 61;
# do python train_retacred.py --model_name_or_path bert-large-cased --input_format typed_entity_marker_pos_punct --seed $SEED --run_name retacred-bert-large-pos;
# done;

# for SEED in 78 23 61;
# do python train_tacred.py --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name tacred-roberta;
# done;

# for SEED in 78 23 61;
# do python train_tacred.py --model_name_or_path roberta-large --input_format typed_entity_marker_pos_punct --seed $SEED --run_name tacred-roberta-pos;
# done;

# for SEED in 78 23 61;
# do python train_retacred.py --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name retacred-roberta;
# done;

# for SEED in 78 23 61;
# do python train_retacred.py --model_name_or_path roberta-large --input_format typed_entity_marker_pos_punct --seed $SEED --run_name retacred-roberta-pos;
# done;