package io.kinference.algorithms.completion.generation.search

import io.kinference.algorithms.completion.generation.*
import java.lang.Math.floorDiv
import java.lang.Math.floorMod
import kotlin.math.*

internal class BeamSearch(
    vocabSize: Int,
    searchSize: Int,
    repetitionPenalty: Double = 1.0
) : Search(vocabSize, searchSize, repetitionPenalty) {

    private var length = 1.0
    override val batchSize: Int
        get() = scores.size
    var scores: MutableList<Double> = arrayListOf(0.0)
        private set

    private var hypotheses: List<MutableList<Int>> = arrayListOf(arrayListOf())
    private var sortMask: IntArray? = null

    override fun step(stepLogProbs: Array<DoubleArray>, context: IntArray): StepResult {
        modifyScore(stepLogProbs, context)

        val stepLogProbsLinearSize = stepLogProbs.sumBy { it.size }
        val logProbs = DoubleArray(stepLogProbsLinearSize)
        val expStepLogProbs = DoubleArray(stepLogProbsLinearSize)
        var offset = 0
        for (i in stepLogProbs.indices) {
            val probs = stepLogProbs[i]
            val score = scores[i]
            for (value in probs) {
                val currentVal = value + score
                logProbs[offset] = currentVal
                expStepLogProbs[offset++] = exp(currentVal)
            }
        }

        var samples = topk1d(logProbs, searchSize)
        val sampleScores = logProbs.sliceArray(samples)

        val stepSortMask = IntArray(samples.size) { floorDiv(samples[it], vocabSize) }
        samples = IntArray(samples.size) { floorMod(samples[it], vocabSize) }

        initSortMask()
        updateState(samples, sampleScores, stepSortMask)
        length += 1

        return StepResult(sortMask!!, samples)
    }

    private fun modifyScore(scores: Array<DoubleArray>, context: IntArray) {
        if (repetitionPenalty != 1.0) {
            val uniqueTokens = context.toSet()
            for (i in scores.indices) {
                pessimizeScore(scores, i, uniqueTokens)
            }

            for (i in hypotheses.indices) {
                pessimizeScore(scores, i, hypotheses[i].toSet())
            }
        }
    }

    private fun pessimizeScore(scores: Array<DoubleArray>, ind: Int, uniqueTokens: Set<Int>) {
        for (previousToken in uniqueTokens) {
            val score = scores[ind][previousToken]
            scores[ind][previousToken] = score * if (score < 0.0) repetitionPenalty else 1.0 / repetitionPenalty
        }
    }

    override fun hypotheses(): List<MutableList<Int>> {
        return hypotheses
    }

    override fun lastPredictions(): IntArray {
        assert(hypotheses.isNotEmpty() && hypotheses[0].size > 0) { "Can't get last predictions if no steps have been performed" }
        return IntArray(hypotheses.size) { hypotheses[it].last() }
    }

    override fun scores(): MutableList<Double> {
        return scores
    }

    private fun initSortMask() {
        sortMask = IntArray(batchSize) { it }
    }

    private fun updateState(samples: IntArray, sampleScores: DoubleArray, sortMask: IntArray) {
        sortState(sortMask)

        scores = sampleScores.toMutableList()
        for (i in hypotheses.indices) {
            hypotheses[i].add(samples[i])
        }
    }

    private fun sortState(sortMask: IntArray? = null) {
        if (sortMask == null) {
            applySliceToState(topk1d(scores.toDoubleArray(), min(searchSize, scores.size)))
        } else {
            applySliceToState(sortMask)
        }
    }

    private fun applySliceToState(tensorSlice: IntArray) {
        scores = scores.slice(tensorSlice).toMutableList()
        hypotheses = tensorSlice.map { ArrayList(hypotheses[it]) }
        if (sortMask != null) {
            sortMask = sortMask!!.sliceArray(tensorSlice)
        }
    }
}
