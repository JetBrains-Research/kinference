package io.kinference.algorithms.completion.generation.search

import io.kinference.algorithms.completion.generation.GenerationInfo
import io.kinference.ndarray.arrays.NDArray


abstract class Search(
    val eosIds: IntArray,
    val vocabSize: Int,
    val searchSize: Int,
    val lenNormBase: Double = 0.0,
    val lenNormPow: Double = 0.0,
    val repetitionPenalty: Double = 1.0
) {

    data class HypothesisInfo(val hypothesis: IntArray, val info: GenerationInfo) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as HypothesisInfo

            if (!hypothesis.contentEquals(other.hypothesis)) return false
            if (info != other.info) return false

            return true
        }

        override fun hashCode(): Int {
            var result = hypothesis.contentHashCode()
            result = 31 * result + info.hashCode()
            return result
        }
    }

    /**
     * Current batch size
     */
    abstract val batchSize: Int

    abstract fun step(stepLogProbs: Array<DoubleArray>, context: IntArray): IntArray

    protected fun stepCheck(logProbs: NDArray) {
        assert(logProbs.shape.contentEquals(intArrayOf(batchSize, vocabSize))) {
            "log_probs must have shape (${batchSize}, $vocabSize), but ${logProbs.shape} was given"
        }

        assert(eosIds.all { it < vocabSize }) {
            "EOS ids must be less than vocab_size, but EOS ids: $eosIds and vocab_size: $vocabSize"
        }
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
