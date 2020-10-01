package io.kinference.generating

import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.model.Model
import io.kinference.ndarray.FloatNDArray
import io.kinference.ndarray.LongNDArray
import io.kinference.ndarray.NDArray
import io.kinference.ndarray.Strides

interface ModelWrapper {
    fun initLogProbs(inputIds: List<Int>): Pair<List<List<MutableList<Double>>>, List<NDArray>>

    fun initLastLogProbs(inputIds: List<Int>): Pair<List<MutableList<Double>>, List<NDArray>> {
        val (score, mems) = initLogProbs(inputIds)
        // (batchSize, seqLen, vocabSize) -> (batchSize, vocabSize)
        val lastProbs: List<MutableList<Double>> = score.map { seq -> seq[seq.size - 1] }
        return Pair(lastProbs, mems)
    }

    fun getLogProbs(inputIds: List<Int>, past: List<NDArray>): Pair<List<MutableList<Double>>, List<NDArray>>
}

class OnnxModelWrapper(modelPath: String) : ModelWrapper {
    val model = Model.load(modelPath)
    private val numAttentionHeads = 4
    private val hiddenSize = 256
    private val numLayer = 3
    private val vocabSize = 50257
//    distilgpt2_l3_h12_d256_int8

    @ExperimentalUnsignedTypes
    override fun initLogProbs(inputIds: List<Int>): Pair<List<List<MutableList<Double>>>, List<NDArray>> {
        val seqLen = inputIds.size
        val input = ArrayList<Tensor>()
        input.add(LongNDArray(inputIds.map { it.toLong() }.toLongArray(), Strides(intArrayOf(1, seqLen))).asTensor("input_ids"))
        input.add(FloatNDArray(List(seqLen) { 1.0F }.toFloatArray(), Strides(intArrayOf(1, seqLen))).asTensor("attention_mask"))
        input.add(LongNDArray(inputIds.indices.map { it.toLong() }.toLongArray(), Strides(intArrayOf(1, seqLen))).asTensor("position_ids"))

        val batchSize = 1
        val shape = intArrayOf(2, batchSize, numAttentionHeads, 0, hiddenSize / numAttentionHeads)
        for (i in 0 until numLayer) {
            val emptyPast = FloatNDArray(floatArrayOf(0.0f), Strides(shape))
            input.add(emptyPast.asTensor("past_$i"))
        }

        val output = model.predict(input)
        val ndProbs = (output[0] as Tensor).data

        val probs: List<List<MutableList<Double>>> = List(batchSize) { List(seqLen) { MutableList(vocabSize) { 0.0 } } }
        for (batch in 0 until batchSize) {
            for (pos in 0 until seqLen) {
                for (id in 0 until vocabSize) {
                    probs[batch][pos][id] = (ndProbs[intArrayOf(batch, pos, id)] as Float).toDouble()
                }
            }
        }

        return Pair(probs, output.subList(1, output.size).map { (it as Tensor).data })
    }

    override fun getLogProbs(inputIds: List<Int>, past: List<NDArray>): Pair<List<MutableList<Double>>, List<NDArray>> {
        TODO("Not yet implemented")
    }
}
