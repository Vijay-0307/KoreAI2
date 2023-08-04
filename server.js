const express = require('express');
const app = express();
const axios = require('axios');
require('@tensorflow/tfjs-backend-webgl');
const tf = require('@tensorflow/tfjs-node');

const modelURL = 'https://teachablemachine.withgoogle.com/models/AJmB8ahNB/model.json'; 
const metadataURL = 'https://teachablemachine.withgoogle.com/models/AJmB8ahNB/metadata.json';

let model;
let metadata;

async function loadModel() {
  model = await tf.loadLayersModel(modelURL);
  metadata = await (await fetch(metadataURL)).json();
}

// Initialize the model on server startup
loadModel().catch(error => {
  console.error('Error loading model:', error);
});

app.get('/predict', async (req, res) => {
  const { imageUrl } = req.query;

  if (!imageUrl) {
    return res.status(400).send('Image URL is required');
  }

  if (!model) {
    return res.status(500).send('Model is not loaded yet. Please wait and try again.');
  }

  try {
    const response = await axios.get(imageUrl, { responseType: 'arraybuffer' });

    if (!response.data) {
      return res.status(400).send('Failed to fetch the image from the provided URL');
    }

    const buffer = Buffer.from(response.data);
    const decodedImage = tf.node.decodeImage(buffer);
    const resizedImage = tf.image.resizeBilinear(decodedImage, [224, 224]);
    const preprocessedImage = resizedImage.div(255.0); // Normalize the pixel values to [0, 1]
    const expandedImage = preprocessedImage.expandDims(0);

    const predictions = model.predict(expandedImage);
    const predictionsArray = Array.from(predictions.dataSync());

    // Map predictions to class labels from metadata
    const classes = metadata?.labels || [];
    const mappedPredictions = predictionsArray.map((confidence, index) => ({
      className: classes[index] || `Class ${index}`,
      confidence,
    }));

    // Filter out predictions with confidence equal to 1
    const confidentPredictions = mappedPredictions.filter((prediction) => prediction.confidence === 1);

    if (confidentPredictions.length === 0) {
      return res.status(404).send('No confident predictions found for the given image');
    }

    // Return the first element with confidence 1
    const confidentPrediction = confidentPredictions[0];
    res.json(confidentPrediction);
  } catch (error) {
    console.error('Error:', error);
    res.status(500).send('An error occurred during prediction');
  }
});

  
  const port = process.env.PORT || 3000;
  app.listen(port, () => {
    console.log(`Server listening on port ${port}!`);
  });