package io.kinference.algorithms.completion.generating

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import kotlin.math.abs

class SearchTest {
    @Test
    @Tag("heavy")
    fun testStep() {
        val search = BeamSearch(intArrayOf(0), vocabSize=5, searchSize=5, lenNormBase=5.0, lenNormPow=0.7)

        val context = IntArray(0)
        val stepLogProbs = arrayOf(doubleArrayOf(0.13, 0.4, 0.17, 0.1, 0.2))
        val sortMask = search.step(stepLogProbs, context)

        assertArrayEquals(sortMask, intArrayOf(0, 0, 0, 0))
        val targetScores1 = doubleArrayOf(0.4000, 0.2000, 0.1700, 0.1000)
        assertTrue(search.scores.mapIndexed { i, d -> abs(d - targetScores1[i]) }.all { it < 1e-5 })

        val sortMask2 = search.step(Array(4) { doubleArrayOf(0.21, 0.11, 0.18, 0.14, 0.36) }, context)
        assertArrayEquals(sortMask2, intArrayOf(0, 0, 1, 0, 2))
        val targetScores2 = doubleArrayOf(0.76, 0.58, 0.56, 0.54, 0.53)
        assertTrue(search.scores.mapIndexed { i, d -> abs(d - targetScores2[i]) }.all { it < 1e-5 })
    }
}
