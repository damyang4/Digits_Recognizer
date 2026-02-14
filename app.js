// Frontend only; predictions happen on the backend API.

function initializeApp() {
  // Create bars dynamically
  createBars();
  
  // Initialize bars
  initializeBars();

  const canvas = document.getElementById("drawingCanvas");
  const ctx = canvas.getContext("2d");

  const previewCanvas = document.getElementById("previewCanvas");
  // const previewCtx = previewCanvas.getContext("2d");

  const clearBtn = document.getElementById("clearBtn");
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = 28;
  tmpCanvas.height = 28;
  const tmpCtx = tmpCanvas.getContext("2d");

  /* SETUP CANVAS */

  // Large canvas for drawing
  canvas.width = 280;
  canvas.height = 280;

  // Small canvas (28x28) for ML
  previewCanvas.width = 28;
  previewCanvas.height = 28;

  // White background
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "black";
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  /* DRAWING LOGIC */

  let isDrawing = false;
  let lastPoint = null;
  let lastPredictTime = 0;
  const PREDICT_THROTTLE_MS = 100; // Tlimit predictions to every 100 ms
  let pendingPredictionFrame = null;
  let isPredicting = false;

  function getCoords(e) {
    const rect = canvas.getBoundingClientRect();

    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }

  function startDrawing(e) {
    isDrawing = true;
    const coords = getCoords(e);
    lastPoint = coords;
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
  }

  function draw(e) {
    if (!isDrawing) return;

    const coords = getCoords(e);
    if (lastPoint) {
      // Using quadratic Bezier curve for smooth curve
      const midX = (lastPoint.x + coords.x) / 2;
      const midY = (lastPoint.y + coords.y) / 2;

      ctx.quadraticCurveTo(lastPoint.x, lastPoint.y, midX, midY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(midX, midY);
    }
    lastPoint = coords;

    // Trigger prediction update
    schedulePredict();
  }

  function stopDrawing() {
    isDrawing = false;
    ctx.closePath();
    lastPoint = null;
  }

  function schedulePredict() {
    // Cancel previous pending prediction (if still queued)
    if (pendingPredictionFrame !== null) {
      cancelAnimationFrame(pendingPredictionFrame);
    }
    
    // Schedule prediction on next animation frame to not block drawing
    pendingPredictionFrame = requestAnimationFrame(() => {
      const now = Date.now();
      if (now - lastPredictTime >= PREDICT_THROTTLE_MS && !isPredicting) {
        lastPredictTime = now;
        isPredicting = true;
        predictDigit().finally(() => {
          isPredicting = false;
        });
      }
      pendingPredictionFrame = null;
    });
  }

  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mouseleave", stopDrawing);


  /* PREDICTION LOGIC */

  function getPixelArray() {
    // Get image data from large canvas (280x280)
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data; // RGBA array for every px [[R, G, B, a],...]

    let minX = canvas.width;
    let minY = canvas.height;
    let maxX = 0;
    let maxY = 0;
    let hasInk = false;

    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const idx = (y * canvas.width + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];

        // luminosity formula to turn it to grayscale
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;

        if (gray < 245) {
          hasInk = true;
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (!hasInk) {
      return new Array(784).fill(0);
    }

    // Scaling logic
    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;

    // Scale to in 20x20, preserving aspect ratio
    const targetSize = 20;
    const scale = targetSize / Math.max(boxWidth, boxHeight);
    const scaledWidth = Math.max(1, Math.round(boxWidth * scale));
    const scaledHeight = Math.max(1, Math.round(boxHeight * scale));

    // Center in 28x28 canvas
    const offsetX = Math.round((28 - scaledWidth) / 2);
    const offsetY = Math.round((28 - scaledHeight) / 2);

    tmpCtx.imageSmoothingEnabled = true;
    tmpCtx.fillStyle = "white";
    tmpCtx.fillRect(0, 0, 28, 28);

    // Copy and scale from large canvas to temp canvas
    tmpCtx.drawImage(
      canvas,
      minX,
      minY,
      boxWidth,
      boxHeight,
      offsetX,
      offsetY,
      scaledWidth,
      scaledHeight
    );

    const normData = tmpCtx.getImageData(0, 0, 28, 28).data;
    const pixels = new Array(784);

    for (let i = 0; i < 784; i++) {
      const idx = i * 4; // 4-bytes per pixel
      const r = normData[idx];
      const g = normData[idx + 1];
      const b = normData[idx + 2];
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      pixels[i] = Math.round(255 - gray);  // invert
    }

    return pixels;
  }

  function euclideanDistance(arr1, arr2) {
    let sum = 0;
    for (let i = 0; i < arr1.length; i++) {
      // Normalize to 0-1 range for more stable distance
      const val1 = arr1[i] / 255;
      const val2 = arr2[i] / 255;
      sum += (val1 - val2) ** 2;
    }
    return Math.sqrt(sum);
  }

  async function predictDigit() {
    const drawnPixels = getPixelArray();

    // Check if canvas is mostly empty
    if (drawnPixels.every(pixel => pixel < 10)) {
      resetBars();
      return;
    }

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixels: drawnPixels })
      });

      if (!response.ok) {
        console.error("Prediction request failed:", response.status);
        return;
      }

      const result = await response.json();
      if (!result || typeof result.topDigit !== "number") {
        return;
      }

      displayPrediction(result.topDigit, result.confidences, result.topConfidence);
      updateBars(result.confidences, result.topDigit);
    } catch (error) {
      console.error("Prediction error:", error);
    }
  }

  function displayPrediction(digit, confidences, confidence) {
    const output = document.getElementById("predictionOutput");
    if (output) {
      const confPercent = Math.round(confidence * 100);
      output.textContent = `${digit} (Confidence: ${confPercent}%)`;
      output.style.color = confidence > 0.6 ? "green" : confidence > 0.3 ? "orange" : "red";
    }
  }

  function resetBars() {
    const output = document.getElementById("predictionOutput");
    if (output) {
      output.textContent = "Draw a digit...";
      output.style.color = "gray";
    }

    initializeBars();
  }

  clearBtn.addEventListener("click", () => {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resetBars();
  });
}

function initializeBars() {
  for (let i = 0; i <= 9; i++) {
    const bar = document.getElementById(`bar${i}`);
    const label = document.getElementById(`label${i}`);
    if (bar) {
      bar.style.height = "0%";
      bar.classList.remove("top-prediction");
    }
    if (label) {
      label.textContent = i;
      label.classList.remove("top-prediction");
    }
  }
}

function updateBars(confidences, topPrediction) {
  for (let digit = 0; digit <= 9; digit++) {
    const bar = document.getElementById(`bar${digit}`);
    const label = document.getElementById(`label${digit}`);
    const confidence = Math.round(confidences[digit] * 100);

    if (bar) {
      bar.style.height = confidence + "%";
      if (digit === topPrediction) {
        bar.classList.add("top-prediction");
      } else {
        bar.classList.remove("top-prediction");
      }
    }
    if (label) {
      label.textContent = digit;
      if (digit === topPrediction) {
        label.classList.add("top-prediction");
      } else {
        label.classList.remove("top-prediction");
      }
    }
  }
}

document.addEventListener("DOMContentLoaded", initializeApp);

function createBars() {
  const container = document.getElementById("barsContainer");
  if (!container) return;
  
  container.innerHTML = ""; // Clear any existing bars
  
  for (let i = 0; i <= 9; i++) {
    const barWrapper = document.createElement("div");
    barWrapper.className = "bar-wrapper";
    
    const label = document.createElement("div");
    label.id = `label${i}`;
    label.className = "bar-label";
    label.textContent = i;
    
    const barTrack = document.createElement("div");
    barTrack.className = "bar-track";
    
    const bar = document.createElement("div");
    bar.id = `bar${i}`;
    bar.className = "bar-fill";
    bar.style.height = "0%";
    
    barTrack.appendChild(bar);
    barWrapper.appendChild(label);
    barWrapper.appendChild(barTrack);
    container.appendChild(barWrapper);
  }
}
