package com.gesture.app

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * OTA (over-the-air) model updates.
 * Fetches version.json from the server, downloads the model if version is newer,
 * and stores it in app files. Version is stored in SharedPreferences for display.
 */
object OtaManager {

    private const val TAG = "OtaManager"
    const val OTA_BASE_URL = "https://tinyml-ota-update.vercel.app"
    private const val PREFS_NAME = "ota_prefs"
    private const val KEY_MODEL_VERSION = "model_version"
    private const val KEY_MODEL_UPDATED = "model_updated"

    private fun prefs(context: Context) =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun getModelFile(context: Context): File {
        val dir = File(context.filesDir, "model").apply { mkdirs() }
        return File(dir, "gesture_model_quant.tflite")
    }

    /** Current stored version (0 = bundled only, no download yet). */
    fun getStoredVersion(context: Context): Int =
        prefs(context).getInt(KEY_MODEL_VERSION, 0)

    /** Human-readable date string when model was last updated (for display). */
    fun getModelUpdatedString(context: Context): String? =
        prefs(context).getString(KEY_MODEL_UPDATED, null)

    /**
     * Fetch version.json and return (version, modelUrl) or null on error.
     */
    fun fetchVersionJson(): Pair<Int, String>? {
        return try {
            val url = URL("$OTA_BASE_URL/version.json")
            val conn = url.openConnection() as? HttpURLConnection ?: return null
            try {
                conn.requestMethod = "GET"
                conn.connectTimeout = 10_000
                conn.readTimeout = 10_000
                conn.connect()
                if (conn.responseCode != 200) return null
                val body = conn.inputStream.bufferedReader().readText()
                val obj = JSONObject(body)
                val version = obj.optInt("version", 0)
                val modelUrl = obj.optString("modelUrl", "/model/gesture_model_quant.tflite")
                Pair(version, modelUrl)
            } finally {
                conn.disconnect()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Fetch version failed", e)
            null
        }
    }

    /**
     * Download model from full URL (BASE + path) and return bytes, or null on error.
     */
    fun downloadModel(modelUrlPath: String): ByteArray? {
        val path = if (modelUrlPath.startsWith("/")) modelUrlPath else "/$modelUrlPath"
        val urlString = "$OTA_BASE_URL$path"
        return try {
            val url = URL(urlString)
            val conn = url.openConnection() as? HttpURLConnection ?: return null
            try {
                conn.requestMethod = "GET"
                conn.connectTimeout = 15_000
                conn.readTimeout = 30_000
                conn.connect()
                if (conn.responseCode != 200) return null
                conn.inputStream.readBytes()
            } finally {
                conn.disconnect()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Download model failed", e)
            null
        }
    }

    /**
     * Check server version; if newer than stored, download model and save. Returns true if updated.
     */
    fun runOtaCheckAndDownload(context: Context): Boolean {
        val (serverVersion, modelUrlPath) = fetchVersionJson() ?: return false
        val stored = getStoredVersion(context)
        if (serverVersion <= stored) return false
        val bytes = downloadModel(modelUrlPath) ?: return false
        if (bytes.isEmpty()) return false
        val file = getModelFile(context)
        file.parentFile?.mkdirs()
        try {
            file.writeBytes(bytes)
        } catch (e: Exception) {
            Log.w(TAG, "Write model file failed", e)
            return false
        }
        prefs(context).edit()
            .putInt(KEY_MODEL_VERSION, serverVersion)
            .putString(KEY_MODEL_UPDATED, java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US).format(java.util.Date()))
            .apply()
        Log.i(TAG, "OTA updated to model v$serverVersion")
        return true
    }

    /**
     * Returns (model bytes to load, version to display). Version 0 = bundled.
     * Never throws for OTA path: if downloaded file is missing/corrupt, falls back to assets.
     */
    fun getModelBytesAndVersion(context: Context): Pair<ByteArray, Int> {
        val file = getModelFile(context)
        if (file.exists()) {
            try {
                val bytes = file.readBytes()
                if (bytes.isNotEmpty()) return Pair(bytes, getStoredVersion(context))
            } catch (e: Exception) {
                Log.w(TAG, "OTA file read failed, using bundled", e)
            }
        }
        val bytes = context.assets.open("gesture_model_quant.tflite").use { it.readBytes() }
        return Pair(bytes, 0)
    }
}
