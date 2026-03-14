package com.gesture.app

import kotlin.math.*

/**
 * Extracts the same 39 features as the Python pipeline:
 * - 12 time-domain: mean, std, min, max per axis (accX, accY, accZ)
 * - 27 spectral: FFT length 16, overlap stride 8, log(1+mag), 9 bins x 3 axes
 */
object FeatureExtractor {

    const val WINDOW_SAMPLES = 125   // 2000 ms @ 62.5 Hz
    const val STRIDE_SAMPLES = 5     // 80 ms
    const val FFT_LENGTH = 16
    const val FFT_STRIDE = 8
    const val NUM_FFT_BINS = 9       // FFT_LENGTH/2 + 1 for real
    const val NUM_FEATURES = 39

    private val timeFeatures = FloatArray(12)
    private val spectralAccum = Array(3) { FloatArray(NUM_FFT_BINS) }
    private val spectralCounts = Array(3) { FloatArray(NUM_FFT_BINS) }

    /**
     * Compute 39 features from a window of shape [125][3] (accX, accY, accZ).
     */
    fun extract(window: Array<FloatArray>): FloatArray {
        require(window.size >= WINDOW_SAMPLES) { "Window must have at least $WINDOW_SAMPLES rows" }
        require(window[0].size >= 3) { "Each row must have 3 values (accX, accY, accZ)" }

        // Time-domain: mean, std, min, max per axis -> 12
        for (axis in 0..2) {
            var sum = 0.0
            var min = Float.MAX_VALUE
            var max = Float.MIN_VALUE
            for (i in 0 until WINDOW_SAMPLES) {
                val v = window[i][axis]
                sum += v
                if (v < min) min = v
                if (v > max) max = v
            }
            val mean = sum / WINDOW_SAMPLES
            var varSum = 0.0
            for (i in 0 until WINDOW_SAMPLES) {
                val d = window[i][axis] - mean
                varSum += d * d
            }
            val std = sqrt(varSum / WINDOW_SAMPLES).toFloat()
            if (std.isNaN()) timeFeatures[axis * 4 + 1] = 0f else timeFeatures[axis * 4 + 1] = std
            timeFeatures[axis * 4] = mean.toFloat()
            timeFeatures[axis * 4 + 2] = min
            timeFeatures[axis * 4 + 3] = max
        }

        // Spectral: FFT length 16, stride 8, log1p(mag), average -> 9 bins x 3 axes = 27
        for (a in 0..2) for (b in 0 until NUM_FFT_BINS) {
            spectralAccum[a][b] = 0f
            spectralCounts[a][b] = 0f
        }
        var frameStart = 0
        while (frameStart + FFT_LENGTH <= WINDOW_SAMPLES) {
            for (axis in 0..2) {
                val frame = FloatArray(FFT_LENGTH) { window[frameStart + it][axis] }
                val mags = rfftMagnitudes(frame)
                for (b in 0 until NUM_FFT_BINS) {
                    spectralAccum[axis][b] += ln(1f + mags[b])
                    spectralCounts[axis][b] += 1f
                }
            }
            frameStart += FFT_STRIDE
        }
        val spectral = FloatArray(27)
        var idx = 0
        for (axis in 0..2)
            for (b in 0 until NUM_FFT_BINS) {
                spectral[idx++] = if (spectralCounts[axis][b] > 0)
                    spectralAccum[axis][b] / spectralCounts[axis][b] else 0f
            }

        return FloatArray(39) { if (it < 12) timeFeatures[it] else spectral[it - 12] }
    }

    /** Real FFT length 16 -> 9 magnitude bins. */
    private fun rfftMagnitudes(frame: FloatArray): FloatArray {
        require(frame.size == 16)
        val real = FloatArray(16) { frame[it] }
        val imag = FloatArray(16) { 0f }
        fft(real, imag, 16, false)
        return FloatArray(9) { i ->
            sqrt(real[i] * real[i] + imag[i] * imag[i])
        }
    }

    private fun fft(real: FloatArray, imag: FloatArray, n: Int, inverse: Boolean) {
        var j = 0
        for (i in 0 until n) {
            if (i < j) {
                var t = real[i]; real[i] = real[j]; real[j] = t
                t = imag[i]; imag[i] = imag[j]; imag[j] = t
            }
            var m = n shr 1
            while (m >= 1 && j >= m) {
                j -= m
                m = m shr 1
            }
            j += m
        }
        var len = 2
        while (len <= n) {
            val angle = (if (inverse) 2 else -2) * PI / len
            val wlenReal = cos(angle).toFloat()
            val wlenImag = sin(angle).toFloat()
            var i = 0
            while (i < n) {
                var wReal = 1f
                var wImag = 0f
                for (k in 0 until (len shr 1)) {
                    val u = real[i + k]
                    val v = imag[i + k]
                    val tReal = real[i + k + (len shr 1)]
                    val tImag = imag[i + k + (len shr 1)]
                    real[i + k] = u + tReal
                    imag[i + k] = v + tImag
                    val twReal = wReal * tReal - wImag * tImag
                    val twImag = wReal * tImag + wImag * tReal
                    real[i + k + (len shr 1)] = u - twReal
                    imag[i + k + (len shr 1)] = v - twImag
                    val nwReal = wReal * wlenReal - wImag * wlenImag
                    val nwImag = wReal * wlenImag + wImag * wlenReal
                    wReal = nwReal
                    wImag = nwImag
                }
                i += len
            }
            len = len shl 1
        }
        if (inverse) {
            val nf = n.toFloat()
            for (i in 0 until n) {
                real.set(i, real[i] / nf)
                imag.set(i, imag[i] / nf)
            }
        }
    }
}
