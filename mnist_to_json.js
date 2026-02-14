// Convert MNIST to digits.json (balanced subset)
const fs = require('fs');
const path = require('path');
const https = require('https');
const zlib = require('zlib');

const MNIST_BASE = 'https://storage.googleapis.com/cvdf-datasets/mnist';
const FILES = {
  images: 'train-images-idx3-ubyte.gz',
  labels: 'train-labels-idx1-ubyte.gz'
};

const OUTPUT_PATH = path.join(__dirname, 'dataset', 'digits.json');
const TMP_DIR = path.join(__dirname, 'dataset', 'mnist_tmp');
const TOTAL_SAMPLES = 60000;
const PER_DIGIT = Math.floor(TOTAL_SAMPLES / 10);

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (res) => {
      if (res.statusCode !== 200) {
        reject(new Error(`Failed to download ${url}. Status: ${res.statusCode}`));
        return;
      }
      res.pipe(file);
      file.on('finish', () => file.close(resolve));
    }).on('error', (err) => {
      fs.unlink(dest, () => reject(err));
    });
  });
}

function gunzipFile(filePath) {
  const data = fs.readFileSync(filePath);
  return zlib.gunzipSync(data);
}
/* IDX MNIST format:
Bytes 0-3:   Magic number (0x00000803 = 2051 in decimal)
Bytes 4-7:   Number of images (0x0000EA60 = 60,000)
Bytes 8-11:  Image rows (0x0000001C = 28)
Bytes 12-15: Image columns (0x0000001C = 28)
*/
function parseImages(buffer) { // parsing IDX MNIST format
  const magic = buffer.readUInt32BE(0);
  if (magic !== 2051) {
    throw new Error(`Invalid images file magic: ${magic}`);
  }
  const count = buffer.readUInt32BE(4);
  const rows = buffer.readUInt32BE(8);
  const cols = buffer.readUInt32BE(12);
  const imageSize = rows * cols;
  const images = [];
  let offset = 16;

  for (let i = 0; i < count; i++) {
    const pixels = new Array(imageSize);
    for (let p = 0; p < imageSize; p++) {
      pixels[p] = buffer[offset + p];
    }
    images.push(pixels);
    offset += imageSize;
  }
  return images; // returns array of 60000 images, each with 28X28 px
}

function parseLabels(buffer) {
  const magic = buffer.readUInt32BE(0);
  if (magic !== 2049) {
    throw new Error(`Invalid labels file magic: ${magic}`);
  }
  const count = buffer.readUInt32BE(4);
  const labels = new Array(count);
  for (let i = 0; i < count; i++) {
    labels[i] = buffer[8 + i];
  }
  return labels;
}

async function main() {
  fs.mkdirSync(TMP_DIR, { recursive: true });

  const imagesPath = path.join(TMP_DIR, FILES.images);
  const labelsPath = path.join(TMP_DIR, FILES.labels);

  console.log('Downloading MNIST files...');
  await downloadFile(`${MNIST_BASE}/${FILES.images}`, imagesPath);
  await downloadFile(`${MNIST_BASE}/${FILES.labels}`, labelsPath);

  console.log('Decompressing...');
  const imagesBuf = gunzipFile(imagesPath);
  const labelsBuf = gunzipFile(labelsPath);

  console.log('Parsing IDX files...');
  const images = parseImages(imagesBuf);
  const labels = parseLabels(labelsBuf);

  const dataset = {};
  const counts = {};
  for (let d = 0; d <= 9; d++) {
    dataset[d] = [];
    counts[d] = 0;
  }

  console.log('Selecting balanced subset...');
  for (let i = 0; i < labels.length; i++) {
    const digit = labels[i];
    if (counts[digit] < PER_DIGIT) { // if digit has fewer than 6000 -> add
      dataset[digit].push({ pixels: images[i] });
      counts[digit]++;
    }
    // stop if all have 6000 samples already
    const done = Object.values(counts).every(c => c >= PER_DIGIT);
    if (done) break;
  }

  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  console.log('Writing digits.json...');
  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(dataset), 'utf8');

  console.log('Done.');
  console.log(`Per digit: ${PER_DIGIT}`);
  console.log(`Total samples: ${total}`);
  console.log(`Saved to: ${OUTPUT_PATH}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
