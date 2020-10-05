package io.kinference.generating

import io.kinference.ndarray.*
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class ModelWrapperTest {
    @ExperimentalUnsignedTypes
    @Test
    @Tag("heavy")
    fun testExecutable() {
        val baseDir = "/Users/aleksandr.khvorov/jb/grazie/grazie-datasets/src"
        val model = OnnxModelWrapper("$baseDir/completion/big/opt/onnxrt/onnx_models/distilgpt2_l3_h12_d256_int8.onnx")
        val result1 = model.initLastLogProbs(listOf(listOf(1, 2, 3, 452)))
        val targetProb1 = -23.7406
        assertTrue(kotlin.math.abs(result1.first[0][234] - targetProb1) < 0.1)

        val result2 = model.getLogProbs(listOf(listOf(578)), result1.second)
        val targetProb2 = -20.3615
        assertTrue(kotlin.math.abs(result2.first[0][0][234] - targetProb2) < 0.3)

        val sortMask = listOf(0, 0)
        val doublePast = result1.second.map { mem ->
            val shape = mem.shape
            val size = mem.linearSize * sortMask.size / shape[1]
            val values: MutableList<Float> = ArrayList(size)
            for (i in 0 until shape[0]) {
                val row = mem.row(i)
                for (j in sortMask.indices) {
                    values.addAll((row.row(sortMask[j]) as FloatNDArray).array.toList())
                }
            }
            MutableFloatNDArray(values.toFloatArray(), Strides(intArrayOf(shape[0], sortMask.size, shape[2], shape[3], shape[4])))
        }

        val result3 = model.getLogProbs(listOf(listOf(578), listOf(23)), doublePast)
        val targetProb30 = -20.3615
        val targetProb31 = -16.4451
        assertTrue(kotlin.math.abs(result3.first[0][0][234] - targetProb30) < 0.3)
        assertTrue(kotlin.math.abs(result3.first[1][0][234] - targetProb31) < 0.3)
    }
}
