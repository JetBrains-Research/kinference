package io.kinference.completion.generating

import io.kinference.completion.model27
import io.kinference.ndarray.*
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class ModelWrapperTest {
    companion object {
        val config = model27
    }

    @ExperimentalUnsignedTypes
    @Test
    @Tag("heavy")
    fun testInit() {
        val model = OnnxModelWrapper(config)
        val result1 = model.initLogProbs(arrayOf(intArrayOf(1, 2, 3, 452)))
        val targetProb1 = -23.7406
        assertTrue(kotlin.math.abs(result1.first[0][result1.first[0].size - 1][234] - targetProb1) < 0.1)
    }

    @ExperimentalUnsignedTypes
    @Test
    @Tag("heavy")
    fun testInitLast() {
        val model = OnnxModelWrapper(config)
        val result1 = model.initLastLogProbs(arrayOf(intArrayOf(1, 2, 3, 452)))
        val targetProb1 = -23.7406
        assertTrue(kotlin.math.abs(result1.first[0][234] - targetProb1) < 0.1)
    }

    @ExperimentalUnsignedTypes
    @Test
    @Tag("heavy")
    fun testGetLogsOne() {
        val model = OnnxModelWrapper(config)
        val result1 = model.initLastLogProbs(arrayOf(intArrayOf(1, 2, 3, 452)))

        val result2 = model.getLogProbs(arrayOf(intArrayOf(578)), result1.second)
        val targetProb2 = -20.3615
        assertTrue(kotlin.math.abs(result2.first[0][0][234] - targetProb2) < 0.3)
    }

    @ExperimentalUnsignedTypes
    @Test
    @Tag("heavy")
    fun testGetLogsFew() {
        val model = OnnxModelWrapper(config)
        val result1 = model.initLastLogProbs(arrayOf(intArrayOf(1, 2, 3, 452)))

        val doublePast = reorderPastStates(result1.second, listOf(0, 0))

        val result3 = model.getLogProbs(arrayOf(intArrayOf(578), intArrayOf(23)), doublePast)
        val targetProb30 = -20.3615
        val targetProb31 = -16.4451
        assertTrue(kotlin.math.abs(result3.first[0][0][234] - targetProb30) < 0.3)
        assertTrue(kotlin.math.abs(result3.first[1][0][234] - targetProb31) < 0.3)
    }

    @ExperimentalUnsignedTypes
    @Test
    @Tag("heavy")
    fun testGetLastLogsFew() {
        val model = OnnxModelWrapper(config)
        val result1 = model.initLastLogProbs(arrayOf(intArrayOf(1, 2, 3, 452)))

        val doublePast = reorderPastStates(result1.second, listOf(0, 0))

        val result4 = model.getLastLogProbs(intArrayOf(578, 23), doublePast)
        val targetProb40 = -20.3615
        val targetProb41 = -16.4451
        assertTrue(kotlin.math.abs(result4.first[0][234] - targetProb40) < 0.3)
        assertTrue(kotlin.math.abs(result4.first[1][234] - targetProb41) < 0.3)
    }

    @ExperimentalUnsignedTypes
    private fun reorderPastStates(pastStates: List<MutableNDArray>, sortMask: List<Int>): List<MutableNDArray> {
        return pastStates.map { mem ->
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
    }
}
