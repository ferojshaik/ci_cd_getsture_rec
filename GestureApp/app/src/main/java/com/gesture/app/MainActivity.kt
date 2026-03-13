package com.gesture.app

import android.graphics.Typeface
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.chip.ChipGroup
import java.util.concurrent.Executors

/**
 * Model was trained on 62.5 Hz data (interval_ms=16). On device, sensor rate varies (~50–200 Hz).
 * We resample to 62.5 Hz so 125 samples always represent 2.0 s, matching training.
 */
class MainActivity : AppCompatActivity(), SensorEventListener {

    companion object {
        private const val TAG = "GestureApp"
    }

    private var sensorManager: SensorManager? = null
    private var classifier: GestureClassifier? = null
    private var labelView: TextView? = null
    private var statusView: TextView? = null
    private var modelVersionView: TextView? = null
    private var chipGroup: ChipGroup? = null
    private var statusDot: android.view.View? = null
    private var scoreIdealView: TextView? = null
    private var scoreTapView: TextView? = null
    private var scoreShakeView: TextView? = null
    private var scoreWaveView: TextView? = null
    private var currentModelVersion: Int = 0

    private val chipIds = intArrayOf(R.id.tap_chip, R.id.wave_chip, R.id.shake_chip, R.id.idle_chip)

    // Time-based buffer: need 2+ s of data at any rate, then resample to 125 @ 62.5 Hz
    private data class Sample(val timestampNs: Long, val x: Float, val y: Float, val z: Float)
    private val timeBuffer = mutableListOf<Sample>()
    private val maxBufferNs = 2_500_000_000L   // 2.5 s in ns
    private val windowDurationNs = 2_000_000_000L  // 2.0 s
    private val sampleIntervalNs = 16_000_000L     // 16 ms = 62.5 Hz

    private val mainHandler = Handler(Looper.getMainLooper())
    private val stableCount = intArrayOf(0)
    private val lastPrediction = intArrayOf(-1)
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private var lastInferenceTimeMs = 0L
    private val inferenceIntervalMs = 80L  // throttle to ~12.5 inferences/sec, avoid ANR

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            setContentView(R.layout.activity_main)
            labelView = findViewById(R.id.label)
            statusView = findViewById(R.id.status)
            modelVersionView = findViewById(R.id.model_version)
            chipGroup = findViewById(R.id.chip_group)
            statusDot = findViewById(R.id.status_dot)
            scoreIdealView = findViewById(R.id.score_ideal)
            scoreTapView = findViewById(R.id.score_tap)
            scoreShakeView = findViewById(R.id.score_shake)
            scoreWaveView = findViewById(R.id.score_wave)
            labelView?.setTextColor(ContextCompat.getColor(this, R.color.listening_tint))
            labelView?.textSize = 36f
            statusView?.text = getString(R.string.status_loading)

            // OTA check + load model on background thread (OTA failure never blocks; we always fall back to bundled)
            Thread {
                try {
                    try {
                        OtaManager.runOtaCheckAndDownload(this)
                    } catch (e: Exception) {
                        Log.w(TAG, "OTA check failed, using bundled model", e)
                    }
                    val (bytes, version) = OtaManager.getModelBytesAndVersion(this)
                    val cl = GestureClassifier(this, bytes)
                    currentModelVersion = version
                    mainHandler.post {
                        classifier = cl
                        updateModelVersionDisplay()
                        onClassifierReady()
                    }
                } catch (e: Throwable) {
                    Log.e(TAG, "Model load failed", e)
                    mainHandler.post {
                        statusView?.text = "Model missing. Rebuild app with model in assets."
                    }
                }
            }.start()
        } catch (e: Throwable) {
            Log.e(TAG, "onCreate failed", e)
            val msg = TextView(this).apply {
                text = "Error: ${e.message}\n\nCheck that the app was built with the model in assets."
                setPadding(48, 48, 48, 48)
            }
            setContentView(msg)
        }
    }

    private fun updateModelVersionDisplay() {
        val v = currentModelVersion
        modelVersionView?.text = if (v == 0) {
            getString(R.string.model_version_bundled)
        } else {
            val updated = OtaManager.getModelUpdatedString(this)
            if (updated != null) getString(R.string.model_version_ota, v, updated)
            else getString(R.string.model_version_v, v)
        }
    }

    private fun onClassifierReady() {
        sensorManager = getSystemService(SENSOR_SERVICE) as? SensorManager
        val acc = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        if (acc == null) {
            statusView?.text = "No accelerometer"
            return
        }
        sensorManager?.registerListener(this, acc, SensorManager.SENSOR_DELAY_FASTEST)
        statusView?.text = getString(R.string.status_active)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null || event.sensor.type != Sensor.TYPE_ACCELEROMETER) return
        if (classifier == null) return
        val ts = event.timestamp
        val x = event.values[0]
        val y = event.values[1]
        val z = event.values[2]
        synchronized(timeBuffer) {
            timeBuffer.add(Sample(ts, x, y, z))
            val cutoff = ts - maxBufferNs
            while (timeBuffer.isNotEmpty() && timeBuffer.first().timestampNs < cutoff) {
                timeBuffer.removeAt(0)
            }
        }
        val now = System.currentTimeMillis()
        if (now - lastInferenceTimeMs < inferenceIntervalMs) return
        lastInferenceTimeMs = now
        val bufferCopy = synchronized(timeBuffer) { timeBuffer.toList() }
        if (bufferCopy.size < 2) return
        val tEnd = bufferCopy.last().timestampNs
        val tStart = tEnd - windowDurationNs
        if (bufferCopy.first().timestampNs > tStart) return
        inferenceExecutor.execute {
            runInference(bufferCopy)
        }
    }

    private fun runInference(bufferCopy: List<Sample>) {
        val cl = classifier ?: return
        try {
            val tEnd = bufferCopy.last().timestampNs
            val tStart = tEnd - windowDurationNs
            val window = Array(FeatureExtractor.WINDOW_SAMPLES) { i ->
                val t = tStart + i * sampleIntervalNs
                interpolate(bufferCopy, t)
            }
            val features = FeatureExtractor.extract(window)
            val scores = cl.classifyWithScores(features)
            val pred = scores.indices.maxByOrNull { scores[it] } ?: 0
            mainHandler.post { updateScoresDisplay(scores) }
            if (pred == lastPrediction[0]) {
                stableCount[0]++
                val required = if (pred == 2) 1 else 2
                if (stableCount[0] >= required) {
                    val label = cl.labels[pred]
                    mainHandler.post { updateGestureUi(label, pred) }
                }
            } else {
                lastPrediction[0] = pred
                stableCount[0] = 1
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Inference error", e)
        }
    }

    /** Linear interpolation: get (x,y,z) at time t from timeBuffer. */
    private fun interpolate(buffer: List<Sample>, t: Long): FloatArray {
        if (buffer.isEmpty()) return floatArrayOf(0f, 0f, 0f)
        if (t <= buffer.first().timestampNs) return floatArrayOf(buffer.first().x, buffer.first().y, buffer.first().z)
        if (t >= buffer.last().timestampNs) return floatArrayOf(buffer.last().x, buffer.last().y, buffer.last().z)
        var i = 0
        while (i < buffer.size - 1 && buffer[i + 1].timestampNs < t) i++
        val a = buffer[i]
        val b = buffer[i + 1]
        val dt = (b.timestampNs - a.timestampNs).toFloat()
        val frac = if (dt <= 0) 1f else ((t - a.timestampNs).toFloat() / dt).coerceIn(0f, 1f)
        return floatArrayOf(
            a.x + frac * (b.x - a.x),
            a.y + frac * (b.y - a.y),
            a.z + frac * (b.z - a.z)
        )
    }

    /** Update the IDEAL | TAP | SHAKE | WAVE confidence row; each value aligned under its label. */
    private fun updateScoresDisplay(scores: FloatArray) {
        if (scores.size < 4) return
        scoreIdealView?.text = scores[3].toInt().coerceIn(0, 100).toString()
        scoreTapView?.text = scores[0].toInt().coerceIn(0, 100).toString()
        scoreShakeView?.text = scores[2].toInt().coerceIn(0, 100).toString()
        scoreWaveView?.text = scores[1].toInt().coerceIn(0, 100).toString()
    }

    private fun updateGestureUi(label: String, index: Int) {
        labelView?.text = label
        labelView?.setTypeface(null, Typeface.BOLD)
        val colorRes = when (index) {
            0 -> R.color.tap_tint
            1 -> R.color.wave_tint
            2 -> R.color.shake_tint
            3 -> R.color.idle_tint
            else -> R.color.text_primary
        }
        labelView?.setTextColor(ContextCompat.getColor(this, colorRes))
        labelView?.textSize = 40f
        if (index in chipIds.indices) {
            chipGroup?.check(chipIds[index])
        }
        // Subtle scale feedback on gesture change
        labelView?.animate()?.scaleX(1.08f)?.scaleY(1.08f)?.setDuration(60)?.withEndAction {
            labelView?.animate()?.scaleX(1f)?.scaleY(1f)?.setDuration(100)?.start()
        }?.start()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onDestroy() {
        inferenceExecutor.shutdown()
        classifier?.close()
        sensorManager?.unregisterListener(this)
        super.onDestroy()
    }
}
