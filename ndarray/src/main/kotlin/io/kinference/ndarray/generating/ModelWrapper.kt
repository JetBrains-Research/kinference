package io.kinference.ndarray.generating

import io.kinference.ndarray.NDArray

interface ModelWrapper {
    fun initLogProbs(input_ids: NDArray): Pair<NDArray, NDArray>

    fun getLogProbs(input_ids: NDArray, past: NDArray): Pair<NDArray, NDArray>
}

class OnnxModelWrapper : ModelWrapper {
    override fun initLogProbs(input_ids: NDArray): Pair<NDArray, NDArray> {
        TODO("Not yet implemented")
    }

    override fun getLogProbs(input_ids: NDArray, past: NDArray): Pair<NDArray, NDArray> {
        TODO("Not yet implemented")
    }
}
