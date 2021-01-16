package io.kinference.algorithms.completion.generation.search

internal abstract class Search(
    val vocabSize: Int,
    val searchSize: Int,
    val repetitionPenalty: Double = 1.0
) {

    internal data class StepResult(val sortMask: IntArray, val newTokens: IntArray)

    /**
     * Current batch size
     */
    abstract val batchSize: Int

    abstract fun step(stepLogProbs: Array<DoubleArray>, context: IntArray): StepResult

    /**
     * List of list of current hypotheses
     */
    abstract fun hypotheses(): List<MutableList<Int>>

    /**
     * Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model
     */
    abstract fun lastPredictions(): IntArray

    /**
     * Scores of hypotheses
     */
    abstract fun scores(): MutableList<Double>
}
