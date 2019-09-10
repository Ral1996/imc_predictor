// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1.85, 1.85, 1.85, 1.85, 1.85, 1.85], [6, 1]);
const ys = tf.tensor2d([80, 81, 83, 85, 81, 82], [6, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 500}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([1.85], [1, 1])).print();
  // Open the browser devtools to see the output
});