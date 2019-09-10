// Define a model for linear regression.
const modelIMC = tf.sequential();
modelIMC.add(tf.layers.dense({units: 1, inputShape: [1]}));

modelIMC.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const heightPatient = tf.tensor2d([1.85, 1.85, 1.85, 1.85, 1.85, 1.85], [6, 1]);
const weightPatient = tf.tensor2d([80, 81, 83, 85, 81, 82], [6, 1]);

// Train the model using the data.
modelIMC.fit(heightPatient, weightPatient, {epochs: 500}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  const predictWeight = modelIMC.predict(tf.tensor2d([1.85], [1, 1])).print();
  // Open the browser devtools to see the output
});
