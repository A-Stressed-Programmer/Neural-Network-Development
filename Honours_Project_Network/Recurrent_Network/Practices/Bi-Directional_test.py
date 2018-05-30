import tensorflow as tf
import numpy as np

# Create input data
X = np.random.randn(2, 10, 8)

# The second example is of length 6 
X[1,6:] = 0
X_lengths = [10, 6]

basic_cell = tf.nn.rnn_cell.LSTMCell(num_units=6)
outputs, states  = tf.contrib.rnn.static_rnn(basic_cell, example_x, dtype=tf.float32)

outputs, states  = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=cell,
    cell_bw=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

output_fw, output_bw = outputs
states_fw, states_bw = states

result = tf.contrib.learn.run_n(
    {"output_fw": output_fw, "output_bw": output_bw, "states_fw": states_fw, "states_bw": states_bw},
    n=1,
    feed_dict=None)

print(result[0]["output_fw"].shape)
print(result[0]["output_bw"].shape)
print(result[0]["states_fw"].h.shape)
print(result[0]["states_bw"].h.shape)
