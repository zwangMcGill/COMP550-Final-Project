# COMP550-Final-Project

The data and results are stored in this google drive:
https://drive.google.com/drive/folders/1DRLew2uZY8Pvh-sSiKbbUDXzUX2n0v8m

**Deriving representations of txt from the NLP model:**

`python extract_nlp_features.py
    --nlp_model [bert/gpt2/xl_net]   
    --sequence_length s
    --output_dir nlp_features`

where s ranges from to 1 to 40. This command derives the representation for all sequences of s consecutive words in the stimuli text in /data/stimuli_words.npy from the model specified in --nlp_model and saves one file for each layer in the model in the specified --output_dir. The names of the saved files contain the argument values that were used to generate them. The output files are numpy arrays of size n_words x n_dimensions, where n_words is the number of words in the stimulus text and n_dimensions is the number of dimensions in the embeddings of the specified model in --nlp_model. Each row of the output file contains the representation of the most recent s consecutive words in the stimulus text (i.e. row i of the output file is derived by passing words i-s+1 to i through the pretrained NLP model).

**Building encoding model to predict fMRI recordings**

`python predict_brain_from_nlp.py
    --subject [F,01]
    --nlp_feat_type [bert/gpt2/xl_net]   
    --nlp_feat_dir INPUT_FEAT_DIR
    --layer l
    --sequence_length s
    --output_dir OUTPUT_DIR`

This call builds encoding models to predict the fMRI recordings using representations of the text stimuli derived from NLP models in step 1 above (INPUT_FEAT_DIR is set to the same directory where the NLP features from step 1 were saved, l and s are the layer and sequence length to be used to load the extracted NLP representations). The encoding model is trained using ridge regression and 4-fold cross validation. The predictions of the encoding model for the heldout data in every fold are saved in an output file in the specified directory OUTPUT_DIR. The output filename is in the following format: predict_{}_with_{}_layer_{}_len_{}.npy, where the first field is specified by --subject, the second by --nlp_feat_type, and the rest by --layer and --sequence_length.

**Evaluating the predictions of the encoding model using classification accuracy**

`python evaluate_brain_predictions.py
    --input_path INPUT_PATH
    --output_path OUTPUT_PATH
    --subject [F,01]`

This call computes the mean 20v20 classification accuracy (over 1000 samplings of 20 words) for each encoding model (from each of the 4 CV folds). The output is a pickle file that contains a list with 4 elements -- one for each CV fold. Each of these 4 elements is another list, which contains the accuracies for all voxels. INPUT_PATH is the full path (including the file name) to the predictions saved in step 2 above. OUTPUT_PATH is the complete path (including file name) to where the accuracies should be saved.
