# #!/bin/bash
# 0shot
python3 transfer_learn_ICL_evi_agnostic.py \
    --k 0\
    --models "TheBloke/Llama-2-70B-Chat-AWQ" \
    --data_path data_csv_file \
    --wiki_S_path pickle_file_labeled_support \
    --wiki_R_path pickle_file_labeled_refute \
    --true_pred_dict_file prediction_file

#1shot
python3 transfer_learn_ICL_evi_agnostic.py \
    --k 1\
    --models "TheBloke/Llama-2-70B-Chat-AWQ" \
    --data_path data_csv_file \
    --wiki_S_path pickle_file_labeled_support \
    --wiki_R_path pickle_file_labeled_refute \
    --true_pred_dict_file prediction_file
