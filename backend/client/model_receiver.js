/**
 * US-09 Subtask 1: Fetch global model from server
 * Retrieves latest aggregated model weights before local training
 */

const SERVER_BASE_URL = 'http://localhost:5000'; // update as per your backend

class ModelReceiver {
  /**
   * Fetch the current global model JSON + weights manifest from server
   */
  static async fetchGlobalModel(modelName = 'global-model') {
    try {
      const startTime = Date.now();

      const response = await fetch(
        `${SERVER_BASE_URL}/models/${modelName}/model.json`
      );

      if (!response.ok) {
        throw new Error(
          `Server returned status ${response.status}`
        );
      }

      const modelJSON = await response.json();

      const elapsed = (Date.now() - startTime) / 1000;
      console.log(
        `✅ Global model fetched in ${elapsed.toFixed(2)}s`
      );

      return {
        success: true,
        modelJSON,
        elapsedSeconds: elapsed
      };
    } catch (error) {
      console.error(
        `❌ Error fetching global model: ${error.message}`
      );
      return { success: false, error: error.message };
    }
  }

  /**
   * Fetch model directly using tf.loadLayersModel (handles weights + topology)
   */
  static async fetchAndLoadModel(modelName = 'global-model') {
    try {
      const startTime = Date.now();

      const modelURL = `${SERVER_BASE_URL}/models/${modelName}/model.json`;
      const model = await tf.loadLayersModel(modelURL);

      const elapsed = (Date.now() - startTime) / 1000;

      if (elapsed > 10) {
        console.warn(
          `⚠️ Distribution took ${elapsed.toFixed(2)}s (exceeds 10s target)`
        );
      } else {
        console.log(
          `✅ Model distributed in ${elapsed.toFixed(2)}s`
        );
      }

      return {
        success: true,
        model,
        elapsedSeconds: elapsed
      };
    } catch (error) {
      console.error(
        `❌ Error loading global model: ${error.message}`
      );
      return { success: false, model: null, error: error.message };
    }
  }

  /**
   * Check server for a new model version before each training round
   */
  static async checkForNewVersion(modelName, currentVersion) {
    try {
      const response = await fetch(
        `${SERVER_BASE_URL}/models/${modelName}/version`
      );
      const data = await response.json();

      return {
        hasNewVersion: data.version !== currentVersion,
        latestVersion: data.version
      };
    } catch (error) {
      console.error(
        `❌ Error checking model version: ${error.message}`
      );
      return { hasNewVersion: false, latestVersion: currentVersion };
    }
  }
}

export { ModelReceiver };