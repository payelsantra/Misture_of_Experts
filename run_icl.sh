# #!/bin/bash
# 0shot
python3 transfer_learn_ICL_evi_agnostic.py \
    --k 0\
    --models "TheBloke/Llama-2-70B-Chat-AWQ" \
    --data_path "/media/pbclab/77a4ead7-6f71-4061-ac44-73526c7285f2/payel/data_open_prompt/data/check_covid/extracted/2_class_non_gopher/check_covid_test_extracted.csv" \
    --wiki_S_path "/media/pbclab/77a4ead7-6f71-4061-ac44-73526c7285f2/payel/data_open_prompt/data/check_covid/extracted/fever_as_training_data_covid_Ext_test/sup_based_lable_top_50_bert_vec_test_covid_Ext_frm_fever.pickle" \
    --wiki_R_path "/media/pbclab/77a4ead7-6f71-4061-ac44-73526c7285f2/payel/data_open_prompt/data/check_covid/extracted/fever_as_training_data_covid_Ext_test/ref_based_lable_top_50_bert_vec_test_covid_Ext_frm_fever.pickle" \
    --true_pred_dict_file "/media/pbclab/77a4ead7-6f71-4061-ac44-73526c7285f2/payel/data_open_prompt/data/check_covid/extracted/predicted_dict/2_class/ICL/predicted_dict_0_shot.pickle"

#1shot
python3 transfer_learn_ICL_evi_agnostic.py \
    --k 1\
    --models "TheBloke/Llama-2-70B-Chat-AWQ" \
    --data_path "/media/pbclab/77a4ead7-6f71-4061-ac44-73526c7285f2/payel/data_open_prompt/data/check_covid/extracted/2_class_non_gopher/check_covid_test_extracted.csv" \
    --wiki_S_path "/media/pbclab/77a4ead7-6f71-4061-ac44-73526c7285f2/payel/data_open_prompt/data/check_covid/extracted/fever_as_training_data_covid_Ext_test/sup_based_lable_top_50_bert_vec_test_covid_Ext_frm_fever.pickle" \
    --wiki_R_path "/media/pbclab/77a4ead7-6f71-4061-ac44-73526c7285f2/payel/data_open_prompt/data/check_covid/extracted/fever_as_training_data_covid_Ext_test/ref_based_lable_top_50_bert_vec_test_covid_Ext_frm_fever.pickle" \
    --true_pred_dict_file "/media/pbclab/77a4ead7-6f71-4061-ac44-73526c7285f2/payel/data_open_prompt/data/check_covid/extracted/predicted_dict/2_class/ICL/predicted_dict_1_shot.pickle"

