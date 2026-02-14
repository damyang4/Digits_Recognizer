const fs = require("fs");
const path = require("path");
const express = require("express");

const app = express();
const PORT = 3000;
const DATASET_PATH = path.join(__dirname, "dataset", "digits.json");

let dataset = null;

function loadDataset() {
  const raw = fs.readFileSync(DATASET_PATH, "utf8");
  dataset = JSON.parse(raw);
  console.log("Dataset loaded:", DATASET_PATH);
}

function euclideanDistance(arr1, arr2) {
  let sum = 0;
  for (let i = 0; i < arr1.length; i++) {
    const val1 = arr1[i] / 255;
    const val2 = arr2[i] / 255;
    sum += (val1 - val2) ** 2;
  }
  return Math.sqrt(sum);
}

function predictDigit(pixels) {
  const allDistances = [];

  for (let digit = 0; digit <= 9; digit++) {
    const samples = dataset[digit];
    if (!samples || !Array.isArray(samples)) continue;

    for (const sample of samples) {
      if (!sample.pixels || !Array.isArray(sample.pixels)) continue;
      const dist = euclideanDistance(pixels, sample.pixels);
      allDistances.push({ digit, distance: dist });
    }
  }

  const k = 15;
  allDistances.sort((a, b) => a.distance - b.distance);
  const neighbors = allDistances.slice(0, k);

  const votes = {};
  const weightedVotes = {};
  for (let i = 0; i <= 9; i++) {
    votes[i] = 0;
    weightedVotes[i] = 0;
  }

  const maxDist = Math.max(...neighbors.map((n) => n.distance));
  const minDist = Math.min(...neighbors.map((n) => n.distance));

  for (const neighbor of neighbors) {
    votes[neighbor.digit]++;
    const weight =
      maxDist === minDist
        ? 1
        : 1 - (neighbor.distance - minDist) / (maxDist - minDist);
    weightedVotes[neighbor.digit] += weight;
  }

  const confidences = {};
  for (let digit = 0; digit <= 9; digit++) {
    const voteScore = votes[digit] / k;
    const weightScore = weightedVotes[digit] / k;
    confidences[digit] = voteScore * 0.5 + weightScore * 0.5;
  }

  let topDigit = 0;
  let topConfidence = confidences[0];
  for (let digit = 1; digit <= 9; digit++) {
    if (confidences[digit] > topConfidence) {
      topConfidence = confidences[digit];
      topDigit = digit;
    }
  }

  return { topDigit, topConfidence, confidences };
}

app.use(express.json({ limit: "1mb" }));
app.use(express.static(__dirname));

app.post("/predict", (req, res) => {
  if (!dataset) {
    res.status(503).json({ error: "Dataset not loaded" });
    return;
  }

  const pixels = req.body && req.body.pixels;
  if (!Array.isArray(pixels) || pixels.length !== 784) {
    res.status(400).json({ error: "Invalid pixels array" });
    return;
  }

  const result = predictDigit(pixels);
  res.json(result);
});

app.listen(PORT, () => {
  loadDataset();
  console.log(`Server running at http://localhost:${PORT}`);
});
