package io.kinference.algorithms.completion.generation.model

import io.kinference.ndarray.arrays.NDArray


internal interface ModelWrapper {
    val maxSeqLen: Int

    fun initLogProbs(inputIds: Array<IntArray>): ModelOutputSeq

    fun initLastLogProbs(inputIds: Array<IntArray>): ModelOutput {
        return initLogProbs(inputIds).lastLogProbs()
    }

    fun getLogProbs(inputIds: Array<IntArray>, past: List<NDArray>): ModelOutputSeq

    fun getLastLogProbs(inputIds: IntArray, past: List<NDArray>): ModelOutput {
        return getLogProbs(Array(inputIds.size) { intArrayOf(inputIds[it]) }, past).lastLogProbs()
    }
}
