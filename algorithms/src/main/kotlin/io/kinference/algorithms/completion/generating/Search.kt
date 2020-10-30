package io.kinference.algorithms.completion.generating

import io.kinference.ndarray.*
import java.lang.Math.floorDiv
import java.lang.Math.floorMod
import kotlin.math.*

data class HypothesisInfo(val hypothesis: IntArray, val info: GenerationInfo)

abstract class Search(val eosIds: IntArray, val vocabSize: Int, val searchSize: Int,
                      val lenNormBase: Double = 0.0, val lenNormPow: Double = 0.0, val repetitionPenalty: Double = 1.0) {

    /**
     * Current batch size
     */
    abstract val batchSize: Int

    abstract fun step(stepLogProbs: Array<DoubleArray>, context: IntArray): IntArray

    protected fun stepCheck(logProbs: NDArray) {
        assert(logProbs.shape.contentEquals(intArrayOf(batchSize, vocabSize))
        ) { "log_probs must have shape (${batchSize}, $vocabSize), but ${logProbs.shape} was given" }

        assert(eosIds.all { it < vocabSize }
        ) { "EOS ids must be less than vocab_size, but EOS ids: $eosIds and vocab_size: $vocabSize" }
    }

    /**
     * List of list of tuples of current hypotheses and theirs scores
     */
    abstract fun maskedHypotheses(mask: BooleanArray): List<List<HypothesisInfo>>

    /**
     * List of list of tuples of current hypotheses and theirs scores
     */
    abstract fun currentHypotheses(): List<List<HypothesisInfo>>

    /**
     * Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model
     */
    abstract fun lastPredictions(): IntArray
}

class BeamSearch(eosIds: IntArray, vocabSize: Int, searchSize: Int,
                 lenNormBase: Double = 0.0, lenNormPow: Double = 0.0, repetitionPenalty: Double = 1.0) :
    Search(eosIds, vocabSize, searchSize, lenNormBase, lenNormPow, repetitionPenalty) {

    private var length = 1.0
    override val batchSize: Int
        get() = scores.size
    var scores: MutableList<Double> = arrayListOf(0.0)
        private set

    private var hypotheses: List<MutableList<Int>> = arrayListOf(arrayListOf())
    private var eachStepProbs: List<MutableList<Double>> = arrayListOf(arrayListOf())
    private var sortMask: IntArray? = null
    private val eosIdsSet: Set<Int> = eosIds.toSet()

    override fun step(stepLogProbs: Array<DoubleArray>, context: IntArray): IntArray {
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

        var samples = topk1d(logProbs, min((1 + eosIds.size) * searchSize, logProbs.size))
        val sampleScores = logProbs.sliceArray(samples)

        val samplesStepLogProbs = expStepLogProbs.sliceArray(samples)
        val stepSortMask = IntArray(samples.size) { floorDiv(samples[it], vocabSize) }
        samples = IntArray(samples.size) { floorMod(samples[it], vocabSize) }

        initSortMask()
        updateState(samples, sampleScores, samplesStepLogProbs, stepSortMask)
        length += 1

        return sortMask!!
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

    override fun maskedHypotheses(mask: BooleanArray): List<List<HypothesisInfo>> {
        val ans = ArrayList<HypothesisInfo>()
        val score = getNormalizedScores().apply { for (i in this.indices) this[i] = exp(this[i]) }
        for (i in hypotheses.indices) {
            if (mask[i]) {
                ans.add(HypothesisInfo(hypotheses[i].toIntArray(), GenerationInfo(eachStepProbs[i], score[i])))
            }
        }

        return listOf(ans)
    }

    override fun currentHypotheses(): List<List<HypothesisInfo>> {
        val score = getNormalizedScores().apply { for (i in this.indices) this[i] = exp(this[i]) }
        val ans = List(hypotheses.size) {
            HypothesisInfo(hypotheses[it].toIntArray(), GenerationInfo(eachStepProbs[it], score[it]))
        }

        return listOf(ans)
    }

    override fun lastPredictions(): IntArray {
        assert(hypotheses.isNotEmpty() && hypotheses[0].size > 0) {"Can't get last predictions if no steps have been performed"}
        return IntArray(hypotheses.size) { hypotheses[it].last() }
    }

    private fun initSortMask() {
        sortMask = IntArray(batchSize) { it }
    }

    private fun updateState(samples: IntArray, sampleScores: DoubleArray, stepLogProbs: DoubleArray, sortMask: IntArray) {
        sortState(sortMask)

        scores = sampleScores.toMutableList()
        for (i in hypotheses.indices) {
            hypotheses[i].add(samples[i])
            eachStepProbs[i].add(stepLogProbs[i])
        }
        stashTerminated(samples)
    }

    private fun stashTerminated(samples: IntArray) {
        val terminated = isSampleTerminates(samples)
        val notTerminatedInds = (terminated.indices).filter { !terminated[it] }.toIntArray()
        applySliceToState(notTerminatedInds)
        sortState()
}

    private fun sortState(sortMask: IntArray? = null) {
        if (sortMask == null) {
            applySliceToState(topk1d(scores.toDoubleArray(), min(searchSize, scores.size)))
        } else {
            applySliceToState(sortMask)
        }
    }

    private fun isSampleTerminates(samples: IntArray): BooleanArray {
        return BooleanArray(samples.size) { samples[it] in eosIdsSet }
    }

    private fun applySliceToState(tensorSlice: IntArray) {
        scores = scores.slice(tensorSlice).toMutableList()
        hypotheses = tensorSlice.map { ArrayList(hypotheses[it]) }
        eachStepProbs = tensorSlice.map { ArrayList(eachStepProbs[it]) }
        if (sortMask != null) {
            sortMask = sortMask!!.sliceArray(tensorSlice)
        }
    }

    private fun getNormalizedScores(): DoubleArray {
        val normFactor = ((lenNormBase + length) / (lenNormBase + 1)).pow(lenNormPow)
        return DoubleArray(scores.size) { scores[it] / normFactor }
    }
}
