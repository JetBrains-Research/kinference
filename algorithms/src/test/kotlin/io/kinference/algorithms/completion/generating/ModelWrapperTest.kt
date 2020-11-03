package io.kinference.algorithms.completion.generating

import io.kinference.algorithms.completion.model27
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.MutableNDArray
import io.kinference.ndarray.arrays.NDArray
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
        val model = GPT2ModelWrapper(config)
        val result1 = model.initLogProbs(arrayOf(intArrayOf(1, 2, 3, 452)))
        val targetProb1 = -23.7406
        assertTrue(kotlin.math.abs(result1.logProbs[0][result1.logProbs[0].size - 1][234] - targetProb1) < 0.1)
    }

    @ExperimentalUnsignedTypes
    @Test
    @Tag("heavy")
    fun testInitLast() {
        val model = GPT2ModelWrapper(config)
        val result1 = model.initLastLogProbs(arrayOf(intArrayOf(1, 2, 3, 452)))
        val targetProb1 = -23.7406
        assertTrue(kotlin.math.abs(result1.logProbs[0][234] - targetProb1) < 0.1)
    }

    @ExperimentalUnsignedTypes
    @Test
    @Tag("heavy")
    fun testGetLogsOne() {
        val model = GPT2ModelWrapper(config)
        val result1 = model.initLastLogProbs(arrayOf(intArrayOf(1, 2, 3, 452)))

        val result2 = model.getLogProbs(arrayOf(intArrayOf(578)), result1.pastStates)
        val targetProb2 = -20.3615
        assertTrue(kotlin.math.abs(result2.logProbs[0][0][234] - targetProb2) < 0.3)
    }

    @ExperimentalUnsignedTypes
    @Test
    @Tag("heavy")
    fun testGetLogsFew() {
        val model = GPT2ModelWrapper(config)
        val result1 = model.initLastLogProbs(arrayOf(intArrayOf(1, 2, 3, 452)))

        val doublePast = reorderPastStates(result1.pastStates, listOf(0, 0))

        val result3 = model.getLogProbs(arrayOf(intArrayOf(578), intArrayOf(23)), doublePast)
        val targetProb30 = -20.3615
        val targetProb31 = -16.4451
        assertTrue(kotlin.math.abs(result3.logProbs[0][0][234] - targetProb30) < 0.3)
        assertTrue(kotlin.math.abs(result3.logProbs[1][0][234] - targetProb31) < 0.3)
    }

    @ExperimentalUnsignedTypes
    @Test
    @Tag("heavy")
    fun testGetLastLogsFew() {
        val model = GPT2ModelWrapper(config)
        val result1 = model.initLastLogProbs(arrayOf(intArrayOf(1, 2, 3, 452)))

        val doublePast = reorderPastStates(result1.pastStates, listOf(0, 0))

        val result4 = model.getLastLogProbs(intArrayOf(578, 23), doublePast)
        val targetProb40 = -20.3615
        val targetProb41 = -16.4451
        assertTrue(kotlin.math.abs(result4.logProbs[0][234] - targetProb40) < 0.3)
        assertTrue(kotlin.math.abs(result4.logProbs[1][234] - targetProb41) < 0.3)
    }

    @ExperimentalUnsignedTypes
    private fun reorderPastStates(pastStates: List<NDArray>, sortMask: List<Int>): List<MutableNDArray> {
        return pastStates.map { mem ->
            val shape = mem.shape
            val size = mem.linearSize * sortMask.size / shape[1]
            val values = ArrayList<FloatArray>(size)
            for (i in 0 until shape[0]) {
                val row = mem.row(i)
                for (j in sortMask.indices) {
                    values.add((row.row(sortMask[j]) as FloatNDArray).array.toArray())
                }
            }
            MutableFloatNDArray(values.reduce(FloatArray::plus), Strides(intArrayOf(shape[0], sortMask.size, shape[2], shape[3], shape[4])))
        }
    }
}
