package io.kinference.algorithms.completion.generation.model

import io.kinference.ndarray.arrays.NDArray

internal data class ModelOutputSeq(val logProbs: List<Array<DoubleArray>> = emptyList(), val pastStates: List<NDArray> = emptyList()) {
    fun lastLogProbs(): ModelOutput {
        val lastProbs: Array<DoubleArray> = Array(logProbs.size) { logProbs[it].last() }
        return ModelOutput(lastProbs, pastStates)
    }
}
