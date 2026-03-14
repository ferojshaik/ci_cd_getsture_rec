package com.gesture.app

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.round

/**
 * Loads gesture_model_quant.tflite and runs inference.
 * Can load from assets (bundled) or from bytes (e.g. OTA-downloaded file).
 * Input: 39 float features -> quantized to int8 using model's scale/zero.
 * Output: 0 = TAP, 1 = WAVE, 2 = SHAKE, 3 = IDLE (IDEAL).
 */
class GestureClassifier(context: Context, modelBytes: ByteArray? = null) {

    private val interpreter: Interpreter
    private val inputScale: Float
    private val inputZeroPoint: Int
    private val outputScale: Float
    private val outputZeroPoint: Int
    private val isQuantized: Boolean

    val labels = arrayOf("TAP", "WAVE", "SHAKE", "IDLE")

    init {
        val model = modelBytes ?: context.assets.open("gesture_model_quant.tflite").use { it.readBytes() }
        val options = Interpreter.Options().setNumThreads(2)
        interpreter = Interpreter(ByteBuffer.allocateDirect(model.size).apply {
            order(ByteOrder.nativeOrder())
            put(model)
            rewind()
        }, options)

        val inputDetails = interpreter.getInputTensor(0)
        val outputDetails = interpreter.getOutputTensor(0)
        val inputQuant = inputDetails.quantizationParams()
        val outputQuant = outputDetails.quantizationParams()
        inputScale = inputQuant.scale
        inputZeroPoint = inputQuant.zeroPoint.toInt()
        outputScale = outputQuant.scale
        outputZeroPoint = outputQuant.zeroPoint.toInt()
        isQuantized = inputDetails.dataType() == org.tensorflow.lite.DataType.INT8
    }

    /**
     * Run inference on 39 features. Returns label index 0=TAP, 1=WAVE, 2=SHAKE, 3=IDLE.
     */
    fun classify(features: FloatArray): Int {
        val scores = classifyWithScores(features)
        return scores.indices.maxByOrNull { scores[it] } ?: 0
    }

    /**
     * Run inference and return 4 class scores as percentages (0–100, sum ≈ 100).
     * Order: [TAP, WAVE, SHAKE, IDLE] (indices 0,1,2,3).
     */
    fun classifyWithScores(features: FloatArray): FloatArray {
        require(features.size == FeatureExtractor.NUM_FEATURES)
        val logits = FloatArray(4)
        if (isQuantized) {
            val inputBuffer = ByteBuffer.allocateDirect(39).order(ByteOrder.nativeOrder())
            for (i in 0 until 39) {
                val q = round(features[i] / inputScale + inputZeroPoint).toInt().coerceIn(-128, 127)
                inputBuffer.put(q.toByte())
            }
            inputBuffer.rewind()
            val outputBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
            interpreter.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()
            for (i in 0 until 4) {
                val q = outputBuffer.get().toInt()
                logits[i] = (q - outputZeroPoint) * outputScale
            }
        } else {
            val inputBuffer = ByteBuffer.allocateDirect(39 * 4).order(ByteOrder.nativeOrder())
            for (f in features) inputBuffer.putFloat(f)
            inputBuffer.rewind()
            interpreter.run(inputBuffer, logits)
        }
        // Softmax and scale to 0–100 for display
        val maxLogit = logits.maxOrNull() ?: 0f
        var sum = 0.0
        for (i in 0 until 4) {
            val e = exp((logits[i] - maxLogit).toDouble())
            sum += e
        }
        return FloatArray(4) { i ->
            (100.0 * exp((logits[i] - maxLogit).toDouble()) / sum).toFloat()
        }
    }

    fun close() = interpreter.close()
}
