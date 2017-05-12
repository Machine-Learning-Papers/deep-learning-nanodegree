# Meets Specifications

Excellent work with the project! Congratulations! :)

I hope the review helped you. If you feel there's something more that you would have preferred from this review please leave a comment. That would immensely help me to improve feedback for any future reviews I conduct including for further projects. Would appreciate your input too. Thanks!

Good luck with the final project!

# Required Files and Tests

##### The project submission contains the project notebook, called “dlnd_language_translation.ipynb”.
##### All the unit tests in project have passed.
All tests passed!

# Preprocessing

The function text_to_ids is implemented correctly.
Great work! Very concise implementation.

# Neural Network

##### The function model_inputs is implemented correctly.

##### The function process_decoding_input is implemented correctly.

Good work!

Now instead of strided_slice() try to see if you can use something like the following -

'cut_piece = target_data[:,:-1]'

Do you think the above does the same and helps?

##### The function encoding_layer is implemented correctly.

Nicely done! Especially for adding dropout!

Here's a question for you -
Do you think dropout should be added to the Basic LSTM cell, like you have, or to the stack of LSTM layers instead?
The function decoding_layer_train is implemented correctly.
Nice work!

**Important**

In your decoding_layer function you have implemented below, you will see that you call this function, thedecoding_layer_train function, by passing dec_cell as an argument. In that function, you have already applied a dropout to obtain cell (or rather 'drop_out`).
So in this function, when you define the following -

`drop_out = tf.contrib.rnn.DropoutWrapper(dec_cell, keep_prob)`

You are basically applying dropout to the layers, then stack up the layers to create the Multi RNN Cell and then add another dropout layer to that cell. Adding dropouts in succession can slow down training by a lot. It's not too noticeable in your case since you have the keep probability value set to quite high. So I would recommend you remove the dropout from here, and instead, if you wish to add dropout at all, apply it to theoutputs_train that you calculate in this function instead. You can check out https://www.tensorflow.org/api_docs/python/tf/nn/dropout to do that.

Since this slows down your training mostly but doesn't seem to affect your results otherwise by much, I am passing this point. But please do keep this in mind.

##### The function decoding_layer_infer is implemented correctly.

Good job!

But, an important point -

This is the inference/prediction function where, as you may already know, we don't apply dropout since dropout is only meant for training and adding it to prediction makes us lose some connections. If you check out the code where the graph is built and the training happens, the output of this particular function is run in the session with keep_prob as 1.0. Which is good. But it also means that since we have explicitly defined this function for inference/prediction, adding a dropout here is not really needed.

I just want to be sure you are aware of the above :)

##### The function decoding_layer is implemented correctly.

Excellent!

Some suggestions -

- You can define your MultiRNNCell (And related) within the decoding_scope itself instead of outside of it.
- You can use decoding_scope.reuse_variables() instead of utilizing with tf.variable_scope("decoding", reuse=True) as decoding_scope:. Makes it slightly more convenient as per me :)
- You can consider initializing your weights and biases for your FC layer as well. Proper initialization of the weights can help with the model convergence too. If you notice your loss and accuracies during training oscillate quite a bit which indicates your model has some trouble converging properly as it's overshooting the cost function minimum. So this might help with that.

##### The function seq2seq_model is implemented correctly.

Nicely done! :clap:

Here are a couple of useful resources which can further help you build an intuitive knowledge on the matter https://indico.io/blog/sequence-modeling-neuralnets-part1/ and https://wp.wwu.edu/blogwolf/2017/02/20/seq2seq/ Do check them out! :)

# Neural Network Training

##### The parameters are set to reasonable numbers.

You did a good job tuning the hyperparameters. Some suggestions -

- You selected a good number of epochs. We should tend to select the number of epochs such that you get a low training loss which reaches kind of a steady-state (not much change in value beyond a point).
- You selected a good batch size. Smaller batch sizes take too long to train. Larger batch sizes speed up the training but can degrade the quality of the model. Here is a useful resource to understand this better - http://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
- You selected a good value for RNN size but I would still recommend you try for a larger value here to see how well it can fit your model. What do you think is better - more RNN size with relatively fewer layers, or smaller RNN size with more layers?
- Currently, if you notice, your validation accuracy matches your training accuracy quite well throughout training and evaluating. At times, the validation accuracy is slightly higher than your training accuracy as well. Usually, that indicates that your model capacity is not high enough (http://cs231n.github.io/neural-networks-3/#accuracy) and you could try making the model larger by increasing some parameters (higher RNN size/layers, embedding sizes and such) Although, it's also dependent on the data (including quality) as well. But try it out.
- Select a learning rate so that your model converges well, and there aren't many oscillations/spikes in your training loss as the training progresses. Those spikes (increase and decrease in value) happen mostly because the model overshoots the cost function minimum and can't converge properly. Currently, your training has some of such spikes. So you can try to lower your learning rate as well to see if you can minimize that.
- Do experiment with lower values of keep_probability as well.

Good work!

##### The project should end with a validation and test accuracy that is at least 90.00%

Excellent work! You got a ~94% validation accuracy when I run your model. Really awesome!

# Language Translation

##### The function sentence_to_seq is implemented correctly.

Well done!

You can also use the get() dictionary method in python instead of using the if-else statement. https://www.tutorialspoint.com/python/dictionary_get.htm

##### The project gets majority of the translation correctly. The translation doesn’t have to be perfect.

Well done! :clap:

I get the following results when I run your model for a sample sentence. Quite correct!

```
Input
  Word Ids:      [136, 90, 123, 80, 42, 221, 94, 145, 215, 90, 61, 209, 124, 140, 154]
  English Words: ['california', 'is', 'never', 'cold', 'during', 'february', ',', 'but', 'it', 'is', 'sometimes', 'freezing', 'in', 'june', '.']

Prediction
  Word Ids:      [327, 15, 99, 6, 341, 142, 164, 50, 199, 108, 304, 19, 145, 281, 142, 278, 70]
  French Words: ['californie', 'ne', 'fait', 'jamais', 'froid', 'en', 'février', ',', 'mais', 'il', 'est', 'parfois', 'le', 'gel', 'en', 'juin', '.']
```

Try it out on the larger dataset now if you can!

Also, one very simple question I would like you to think about - If you wanted to translate from French To English, would you have to retrain the entire model with french dataset? Or you think you could somehow reverse the prediction only?