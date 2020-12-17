package io.kinference.algorithms.completion.generation.search

import io.kinference.algorithms.completion.generation.GenerationInfo


internal abstract class Search(
    val eosIds: IntArray,
    val vocabSize: Int,
    val searchSize: Int,
    val lenNormBase: Double = 0.0,
    val lenNormPow: Double = 0.0,
    val repetitionPenalty: Double = 1.0
) {

    /**
     * Current batch size
     */
    abstract val batchSize: Int

    abstract fun step(stepLogProbs: Array<DoubleArray>, context: IntArray): IntArray

    /**
     * List of list of tuples of current hypotheses and theirs scores
     */
    abstract fun currentHypotheses(): List<List<GenerationInfo>>

    /**
     * Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model
     */
    abstract fun lastPredictions(): IntArray
}
