from subprocess import run
# extract features
for i in range(1, 6):
    if i == 0:
        i+=1
    string_to_execute = "python extract_nlp_features.py --nlp_model bert --sequence_length " + str(i) + " --output_dir nlp_features/ "

    run(string_to_execute.split())

# predict brain from nlp
for layer in range(0, 13):
    for seq_len in range(1, 6):
        if seq_len == 0:
            seq_len+=1
        string_to_execute = "python predict_brain_from_nlp.py --subject 01 --nlp_feat_type bert --nlp_feat_dir nlp_features/ --layer " + str(layer) + "  --sequence_length "  + str(seq_len) + " --output_dir ./nlp_features/predictions/ "
        run(string_to_execute.split())

# evaluate predictions
for layer in range(0, 13):
    for seq_len in range(1,6):
        if seq_len == 0:
            seq_len+=1
        string_to_execute = "python evaluate_brain_predictions.py --input_path ./nlp_features/predictions//predict_01_with_bert_layer_" + str(layer) + "_len_" + str(seq_len) + ".npy --output_path ./nlp_features/evaluations/" + str(layer) + "_Len_" + str(seq_len) + " --subject 01 "
        run(string_to_execute.split())



