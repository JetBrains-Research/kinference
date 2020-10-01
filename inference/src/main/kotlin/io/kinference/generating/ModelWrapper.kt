package io.kinference.generating

import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.model.Model
import io.kinference.ndarray.FloatNDArray
import io.kinference.ndarray.LongNDArray
import io.kinference.ndarray.NDArray
import io.kinference.ndarray.Strides
import java.lang.Integer.max

interface ModelWrapper {
    fun initLogProbs(inputIds: List<Int>): Pair<List<List<MutableList<Double>>>, List<NDArray>>

    fun initLastLogProbs(inputIds: List<Int>): Pair<List<MutableList<Double>>, List<NDArray>> {
        val (score, mems) = initLogProbs(inputIds)
        return Pair(score.map { it[it.size - 1] }, mems)
    }

    fun getLogProbs(inputIds: List<Int>, past: List<NDArray>): Pair<List<MutableList<Double>>, List<NDArray>>
}

class OnnxModelWrapper(modelPath: String) : ModelWrapper {
    val model = Model.load(modelPath)
    val numAttentionHeads = 12
    val hiddenSize = 768
    val numLayer = 6

    @ExperimentalUnsignedTypes
    override fun initLogProbs(inputIdsList: List<Int>): Pair<List<List<MutableList<Double>>>, List<NDArray>> {
        val input = ArrayList<Tensor>()
        input.add(LongNDArray(inputIdsList.map { it.toLong() }.toLongArray()).asTensor("input_ids"))
        input.add(FloatNDArray(List(inputIdsList.size) { 1.0F }.toFloatArray()).asTensor("attention_mask"))
        input.add(LongNDArray(inputIdsList.indices.map { it.toLong() }.toLongArray()).asTensor("position_ids"))

        val batchSize = 1
        val shape = intArrayOf(2, batchSize, numAttentionHeads, 0, hiddenSize / numAttentionHeads)
        for (i in 0 until numLayer) {
            val emptyPast = FloatNDArray(floatArrayOf(0.0f), Strides(shape))
            input.add(emptyPast.asTensor("past_$i"))
        }

        val output = model.predict(input)
        val ndProbs = (output[0] as Tensor).data

        val probs = ndProbs as List<List<MutableList<Double>>>  // TODO: convert to lists

        return Pair(probs, output.subList(1, output.size).map { (it as Tensor).data })
    }

    override fun getLogProbs(inputIds: List<Int>, past: List<NDArray>): Pair<List<MutableList<Double>>, List<NDArray>> {
        TODO("Not yet implemented")
    }
}
