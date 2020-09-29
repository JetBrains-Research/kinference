package io.kinference.ndarray.generating

import io.kinference.ndarray.NDArray

interface ModelWrapper {
    fun initLogProbs(inputIds: List<Int>): Pair<List<List<MutableList<Double>>>, NDArray>

    fun initLastLogProbs(inputIds: List<Int>): Pair<List<MutableList<Double>>, NDArray> {
        val (score, mems) = initLogProbs(inputIds)
        return Pair(score.map { it[it.size - 1] }, mems)
    }

    fun getLogProbs(inputIds: List<Int>, past: NDArray): Pair<List<MutableList<Double>>, NDArray>
}

class OnnxModelWrapper : ModelWrapper {
    override fun initLogProbs(inputIds: List<Int>): Pair<List<List<MutableList<Double>>>, NDArray> {
        TODO("Not yet implemented")
    }

    override fun getLogProbs(inputIds: List<Int>, past: NDArray): Pair<List<MutableList<Double>>, NDArray> {
        TODO("Not yet implemented")
    }
}
