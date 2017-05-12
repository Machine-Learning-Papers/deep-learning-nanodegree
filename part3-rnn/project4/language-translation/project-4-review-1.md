# Requires Changes

1 SPECIFICATION REQUIRES CHANGES

It's an outstanding first submission. Almost everything is perfect. A few minor issues are there. By delving a bit more into the project, you will be able to enhance the model's quality.

Keep up the great work!

# Required Files and Tests

The project submission contains the project notebook, called “dlnd_language_translation.ipynb”.
The project submission contains all the required files:

IPython notebook: dlnd_language_translation.ipynb
HTML fule: dlnd_language_translation.html
helper.py
problem_unittests.py
All the unit tests in project have passed.
The unit tests are perfectly running. Unit tests are used for testing the code. Sometimes students face issues understanding the difference between debugging and unit tests. This stackoverflow answer will help you understanding the difference - > http://stackoverflow.com/a/3846212

# Preprocessing

The function text_to_ids is implemented correctly.
The preprocessing of data by converting text to numbers is correctly done using text_to_ids() method. You have correctly added the <EOS> to the end of target sentences.

Suggestion:

The target_id_text expression is a bit long. You can shorten it by specifying an EOS variable like this:

EOS_var = target_vocab_to_int['<EOS>']

source_id_text = [[source_vocab_to_int[word] for word in line.split()] for line in source_text.split('\n')]

target_id_text = [[target_vocab_to_int[word] for word in line.split()] + [EOS_var] for line in target_text.split('\n')]

return source_id_text, target_id_text

# Neural Network

The function model_inputs is implemented correctly.
All the placeholders are precisely created for the Neural Network. These are the main building blocks of NN.
The function process_decoding_input is implemented correctly.
Amazing! It is flawlessly running. The process_decoding_input() method is removing the last word id from each batch in target_data and concatenating the GO ID to the beginning of each batch.
The function encoding_layer is implemented correctly.
The encoding_layer() method is correctly creating Encoder RNN layer. The encoder network maps the input sequence to an encoded representation of the sequence. To further read about Encoder-Decoder models in RNN, this quora discussion can be helpful. https://goo.gl/XvrybG

Thumbs up :+1: for using Dropout. The dropout mainly helps to prevent the Neural Nets from Overfitting. . You can further read about Dropouts using this stackoverflow discussion. http://stackoverflow.com/questions/34597316/why-input-is-scaled-in-tf-nn-dropout-in-tensorflow
The function decoding_layer_train is implemented correctly.
The decoding_layer_train() method is correctly method. It's appreciable that you have used the Dropout. :ok_hand:
The function decoding_layer_infer is implemented correctly.
The implementation of decoding_layer_infer() is acceptable. Although it's recommended not to use dropout while creating inference logits. It won't effect either since udacity has hard coded the value of keep_probab to 1.

batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})
The function decoding_layer is implemented correctly.
The Decoder RNN layer is correctly created.

The RNN cell for decoding is precisely created.
The output function is correctly created using the lambda function. :+1:
The decoding_layer_train & decoding_layer_infer functions are correctly used to get training and inference logits.
You can consider reading this article on Deep learning vs inference.
The function seq2seq_model is implemented correctly.
You nailed it! The embedding of the input data, encoding of input data, processing of target data, embedding of target data and decoding of the encoded input are implemented perfectly. :clap:

This Quora discussion can help you further: https://www.quora.com/What-is-conditioning-in-seq2seq-learning

# Neural Network Training

The parameters are set to reasonable numbers.
The hyperparameters are well tuned. You can further enhance its quality. Try using lower values batch_size and embedding for the encoder & decoder.
I highly encourage you dropping the data by decreasing the value of keep_probab. Try to experiment with values like 0.5, 0.6, 0.7. You can also use an automated hyperparameter tuning approach like gridsearch etc.
The project should end with a validation and test accuracy that is at least 90.00%
The project is giving amazing results. The accuracy is over 90%. :clap:

# Language Translation

The function sentence_to_seq is implemented correctly.
All the data is in the text format. The Network interprets data in the form of numbers. Every letter either capital or lower case letter have different ids. That's why it was suggested to first convert the data to lower case. You have missed converting the data to lower case.
This is a minor mistake. Please follow the approach:

- Convert the sentence to lowercase
- Convert words into ids using vocab_to_int
- Convert words not in the vocabulary, to the <UNK> word id.

The project gets majority of the translation correctly. The translation doesn’t have to be perfect.
The translation of good. It still can be improved. After making the above changes in hyperparameter tuning then it may improve the translation.