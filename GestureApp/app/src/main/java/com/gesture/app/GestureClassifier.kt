package com.gesture.app

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
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
     * Run inference on 39 features. Returns label index 0, 1, or 2.
     */
    fun classify(features: FloatArray): Int {
        require(features.size == FeatureExtractor.NUM_FEATURES)
        return if (isQuantized) {
            val inputBuffer = ByteBuffer.allocateDirect(39).order(ByteOrder.nativeOrder())
            for (i in 0 until 39) {
                val q = round(features[i] / inputScale + inputZeroPoint).toInt().coerceIn(-128, 127)
                inputBuffer.put(q.toByte())
            }
            inputBuffer.rewind()
            val outputBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
            interpreter.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()
            var maxIdx = 0
            var maxVal = Float.MIN_VALUE
            for (i in 0 until 4) {
                val q = outputBuffer.get().toInt()
                val dequant = (q - outputZeroPoint) * outputScale
                if (dequant > maxVal) {
                    maxVal = dequant
                    maxIdx = i
                }
            }
            maxIdx
        } else {
            val inputBuffer = ByteBuffer.allocateDirect(39 * 4).order(ByteOrder.nativeOrder())
            for (f in features) inputBuffer.putFloat(f)
            inputBuffer.rewind()
            val outputBuffer = FloatArray(4)
            interpreter.run(inputBuffer, outputBuffer)
            outputBuffer.indices.maxByOrNull { outputBuffer[it] } ?: 0
        }
    }

    fun close() = interpreter.close()
}
