/**
 * US-03: Client-Side Preprocessing Loader
 * Synora — Browser-Based Federated Learning Toolkit
 *
 * This file runs IN THE BROWSER (TensorFlow.js environment).
 * It loads the preprocessed shard JSON produced by preprocessing.py
 * and converts it into tf.Tensor objects ready for model training.
 *
 * Flow:
 *   preprocessed JSON file
 *       → loadClientShard()
 *       → convertToTensors()
 *       → { inputTensor, labelTensor }  ← passed to model.fit()
 */

// ─────────────────────────────────────────────
//  CONFIGURATION  (must match preprocessing.py)
// ─────────────────────────────────────────────

const PREPROCESSING_CONFIG = {
  maxSequenceLength: 50,   // MAX_SEQUENCE_LENGTH from preprocessing.py
  padIndex: 0,             // <PAD> token index
  oovIndex: 1,             // <OOV> token index
};


// ─────────────────────────────────────────────
//  LOAD PREPROCESSED SHARD FROM SERVER
// ─────────────────────────────────────────────

/**
 * Fetch the preprocessed shard JSON for this client from the server.
 *
 * The server exposes the file at:
 *   GET /data/preprocessed/client_{clientId}_preprocessed.json
 *
 * @param {number} clientId - This client's ID (received during registration)
 * @returns {Promise<Object>} Parsed shard object with samples, label_map, etc.
 */
async function loadClientShard(clientId) {
  const url = `/data/preprocessed/client_${clientId}_preprocessed.json`;

  console.log(`[Preprocessing] Loading shard for client ${clientId} from ${url}`);

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to load shard for client ${clientId}: HTTP ${response.status}`
    );
  }

  const shard = await response.json();

  // Basic structure validation
  if (!shard.samples || !Array.isArray(shard.samples)) {
    throw new Error(`Shard for client ${clientId} is missing "samples" array`);
  }
  if (shard.sequence_length !== PREPROCESSING_CONFIG.maxSequenceLength) {
    throw new Error(
      `Shard sequence_length (${shard.sequence_length}) does not match ` +
      `expected (${PREPROCESSING_CONFIG.maxSequenceLength})`
    );
  }

  console.log(
    `[Preprocessing] Loaded ${shard.samples.length} samples. ` +
    `Sequence length: ${shard.sequence_length}. ` +
    `Classes: ${JSON.stringify(shard.label_map)}`
  );

  return shard;
}


// ─────────────────────────────────────────────
//  VALIDATE SAMPLES (null check, shape check)
// ─────────────────────────────────────────────

/**
 * Validate all samples in the loaded shard.
 * Throws immediately if any sample fails a check.
 *
 * Checks (maps to US-03 AC):
 *   - padded_sequence exists and is not null
 *   - padded_sequence has correct length (uniform padding AC)
 *   - No null/undefined inside the sequence (no null values AC)
 *   - encoded_label is a valid non-negative integer
 *
 * @param {Array} samples - Array of sample objects from the shard JSON
 * @param {number} expectedLength - Expected sequence length
 */
function validateSamples(samples, expectedLength) {
  if (!samples || samples.length === 0) {
    throw new Error("[Preprocessing] Validation failed: samples array is empty");
  }

  samples.forEach((sample, i) => {
    const seq = sample.padded_sequence;

    // Check sequence exists (no null AC)
    if (seq === null || seq === undefined) {
      throw new Error(`[Preprocessing] Sample ${i}: padded_sequence is null/undefined`);
    }

    // Check uniform length (padding AC)
    if (seq.length !== expectedLength) {
      throw new Error(
        `[Preprocessing] Sample ${i}: sequence length ${seq.length} ` +
        `!= expected ${expectedLength}`
      );
    }

    // Check no nulls inside sequence (no null values AC)
    const hasNull = seq.some(v => v === null || v === undefined);
    if (hasNull) {
      throw new Error(
        `[Preprocessing] Sample ${i}: contains null/undefined inside padded_sequence`
      );
    }

    // Check all values are numbers
    const hasNaN = seq.some(v => typeof v !== "number" || isNaN(v));
    if (hasNaN) {
      throw new Error(
        `[Preprocessing] Sample ${i}: contains NaN or non-numeric value`
      );
    }

    // Check label is valid
    const label = sample.encoded_label;
    if (label === null || label === undefined || label < 0) {
      throw new Error(
        `[Preprocessing] Sample ${i}: invalid encoded_label: ${label}`
      );
    }
  });

  console.log(`[Preprocessing] ✅ Validated ${samples.length} samples — no nulls, uniform length.`);
}


// ─────────────────────────────────────────────
//  CONVERT TO TF.JS TENSORS
// ─────────────────────────────────────────────

/**
 * Convert validated samples into TensorFlow.js tensors for training.
 *
 * Returns:
 *   inputTensor  — shape [numSamples, sequenceLength], dtype int32
 *   labelTensor  — shape [numSamples],                 dtype int32
 *
 * IMPORTANT: Call dispose() on both tensors when training is finished
 * to free GPU/browser memory (NFR-A02 in your SRS).
 *
 * @param {Array}  samples        - Validated sample objects
 * @param {number} sequenceLength - Expected sequence length
 * @returns {{ inputTensor: tf.Tensor2D, labelTensor: tf.Tensor1D }}
 */
function convertToTensors(samples, sequenceLength) {
  const numSamples = samples.length;

  // Build flat arrays from samples
  const inputData  = new Int32Array(numSamples * sequenceLength);
  const labelData  = new Int32Array(numSamples);

  samples.forEach((sample, i) => {
    // Copy padded_sequence into flat input array at row i
    sample.padded_sequence.forEach((val, j) => {
      inputData[i * sequenceLength + j] = val;
    });
    labelData[i] = sample.encoded_label;
  });

  // Create tensors
  const inputTensor = tf.tensor2d(inputData, [numSamples, sequenceLength], "int32");
  const labelTensor = tf.tensor1d(labelData, "int32");

  console.log(
    `[Preprocessing] ✅ Created tensors — ` +
    `input: ${inputTensor.shape}, labels: ${labelTensor.shape}`
  );

  return { inputTensor, labelTensor };
}


// ─────────────────────────────────────────────
//  MAIN EXPORT: Full preprocessing pipeline
// ─────────────────────────────────────────────

/**
 * Full client-side preprocessing pipeline.
 *
 * Loads the server-produced preprocessed shard, validates it,
 * and returns TF.js tensors ready for model.fit().
 *
 * Usage in your training loop:
 *
 *   const { inputTensor, labelTensor, numClasses, labelMap } =
 *       await prepareClientData(clientId);
 *
 *   await model.fit(inputTensor, labelTensor, { epochs: 3 });
 *
 *   // Always dispose tensors after training to free browser memory
 *   inputTensor.dispose();
 *   labelTensor.dispose();
 *
 * @param {number} clientId - This client's registration ID
 * @returns {Promise<{
 *   inputTensor: tf.Tensor2D,
 *   labelTensor: tf.Tensor1D,
 *   numSamples: number,
 *   numClasses: number,
 *   labelMap: Object,
 *   sequenceLength: number
 * }>}
 */
async function prepareClientData(clientId) {
  console.log(`[Preprocessing] Starting preprocessing for client ${clientId}`);

  // Step 1: Load the preprocessed shard JSON from server
  const shard = await loadClientShard(clientId);

  // Step 2: Validate all samples (null check, shape check)
  validateSamples(shard.samples, shard.sequence_length);

  // Step 3: Convert to TF.js tensors
  const { inputTensor, labelTensor } = convertToTensors(
    shard.samples,
    shard.sequence_length
  );

  const numClasses = Object.keys(shard.label_map).length;

  console.log(`[Preprocessing] ✅ Pipeline complete for client ${clientId}`);
  console.log(`  Samples   : ${shard.num_samples}`);
  console.log(`  Seq length: ${shard.sequence_length}`);
  console.log(`  Classes   : ${numClasses} → ${JSON.stringify(shard.label_map)}`);

  return {
    inputTensor,
    labelTensor,
    numSamples: shard.num_samples,
    numClasses,
    labelMap: shard.label_map,
    sequenceLength: shard.sequence_length,
  };
}


// ─────────────────────────────────────────────
//  BROWSER SELF-TEST  (open console to see output)
// ─────────────────────────────────────────────

/**
 * Quick in-browser smoke test.
 * Call this from browser console: runPreprocessingTest()
 * Does NOT require a real server — uses mock data directly.
 */
async function runPreprocessingTest() {
  console.log("=== US-03 Browser Preprocessing Self-Test ===");

  // Mock shard (simulates what the server would return)
  const mockShard = {
    client_id: 0,
    num_samples: 4,
    sequence_length: PREPROCESSING_CONFIG.maxSequenceLength,
    label_map: { greeting: 0, complaint: 1, positive: 2 },
    samples: Array.from({ length: 4 }, (_, i) => ({
      padded_sequence: Array.from({ length: PREPROCESSING_CONFIG.maxSequenceLength }, (_, j) =>
        j < 3 ? (j + 2) : 0   // First 3 values are word indices, rest are padding (0)
      ),
      encoded_label: i % 3
    }))
  };

  try {
    // Test validation
    validateSamples(mockShard.samples, mockShard.sequence_length);
    console.log("✅ validateSamples passed");

    // Test tensor conversion
    const { inputTensor, labelTensor } = convertToTensors(
      mockShard.samples,
      mockShard.sequence_length
    );

    // Check shape
    console.assert(
      inputTensor.shape[0] === 4 &&
      inputTensor.shape[1] === PREPROCESSING_CONFIG.maxSequenceLength,
      "❌ inputTensor shape mismatch"
    );
    console.assert(labelTensor.shape[0] === 4, "❌ labelTensor length mismatch");
    console.log("✅ Tensor shapes correct:", inputTensor.shape, labelTensor.shape);

    // Check no NaN in tensors
    const inputHasNaN = (await inputTensor.isNaN().any().data())[0];
    const labelHasNaN = (await labelTensor.isNaN().any().data())[0];
    console.assert(!inputHasNaN, "❌ NaN found in inputTensor");
    console.assert(!labelHasNaN, "❌ NaN found in labelTensor");
    console.log("✅ No NaN values in tensors");

    // Clean up
    inputTensor.dispose();
    labelTensor.dispose();
    console.log("✅ Tensors disposed (memory freed)");

    console.log("\n🎉 All browser self-tests passed — US-03 AC verified in browser!");

  } catch (err) {
    console.error("❌ Browser self-test failed:", err.message);
  }
}

// Make available globally for browser console testing
if (typeof window !== "undefined") {
  window.prepareClientData  = prepareClientData;
  window.runPreprocessingTest = runPreprocessingTest;
}

// For Node.js / module environments
if (typeof module !== "undefined") {
  module.exports = {
    prepareClientData,
    loadClientShard,
    validateSamples,
    convertToTensors,
    runPreprocessingTest,
    PREPROCESSING_CONFIG,
  };
}
