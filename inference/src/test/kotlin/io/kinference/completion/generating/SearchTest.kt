package io.kinference.completion.generating

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import kotlin.math.abs

class SearchTest {
    @Test
    @Tag("heavy")
    fun testStep() {
        val search = BeamSearch(intArrayOf(0), vocabSize=5, searchSize=5, lenNormBase=5.0, lenNormPow=0.7)

        val context = emptyList<Int>()
        val stepLogProbs = listOf(mutableListOf(0.13, 0.4, 0.17, 0.1, 0.2))
        val sortMask = search.step(stepLogProbs, context)

        assertEquals(sortMask, listOf(0, 0, 0, 0))
        val targetScores1 = listOf(0.4000, 0.2000, 0.1700, 0.1000)
        assertTrue(search.scores.mapIndexed { i, d -> abs(d - targetScores1[i]) }.all { it < 1e-5 })

        val sortMask2 = search.step(List(4) { mutableListOf(0.21, 0.11, 0.18, 0.14, 0.36) }, context)
        assertEquals(sortMask2, listOf(0, 0, 1, 0, 2))
        val targetScores2 = listOf(0.76, 0.58, 0.56, 0.54, 0.53)
        assertTrue(search.scores.mapIndexed { i, d -> abs(d - targetScores2[i]) }.all { it < 1e-5 })
    }
}
