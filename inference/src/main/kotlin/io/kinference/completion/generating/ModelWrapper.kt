package io.kinference.completion.generating

import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.model.Model
import io.kinference.ndarray.*

interface ModelWrapper {
    fun initLogProbs(inputIds: List<List<Int>>): Pair<List<List<MutableList<Double>>>, List<MutableNDArray>>

    fun initLastLogProbs(inputIds: List<List<Int>>): Pair<List<MutableList<Double>>, List<MutableNDArray>> {
        val (score, mems) = initLogProbs(inputIds)
        // (batchSize, seqLen, vocabSize) -> (batchSize, vocabSize)
        val lastProbs: List<MutableList<Double>> = score.map { seq -> seq[seq.size - 1] }
        return Pair(lastProbs, mems)
    }

    fun getLogProbs(inputIds: List<List<Int>>, past: List<MutableNDArray>): Pair<List<List<MutableList<Double>>>, List<MutableNDArray>>

    fun getLastLogProbs(inputIds: List<Int>, past: List<MutableNDArray>): Pair<List<MutableList<Double>>, List<MutableNDArray>> {
        val (score, mems) = getLogProbs(inputIds.map { listOf(it) }, past)
        // (batchSize, seqLen, vocabSize) -> (batchSize, vocabSize)
        val lastProbs: List<MutableList<Double>> = score.map { seq -> seq[seq.size - 1] }
        return Pair(lastProbs, mems)
    }
}

class OnnxModelWrapper(modelPath: String) : ModelWrapper {
    val model = Model.load(modelPath)
    private val numAttentionHeads = 4
    private val hiddenSize = 256
    private val numLayer = 3
    private val vocabSize = 50257
//    distilgpt2_l3_h12_d256_int8

    @ExperimentalUnsignedTypes
    override fun initLogProbs(inputIds: List<List<Int>>): Pair<List<List<MutableList<Double>>>, List<MutableNDArray>> {
        val batchSize = inputIds.size
        if (batchSize == 0) {
            return Pair(emptyList(), emptyList())
        }

        val seqLen = inputIds[0].size
        val input = ArrayList<Tensor>()
        input.add(LongNDArray(inputIds.flatten().map { it.toLong() }.toLongArray(), Strides(intArrayOf(batchSize, seqLen))).asTensor("input_ids"))
        input.add(FloatNDArray(List(batchSize * seqLen) { 1.0F }.toFloatArray(), Strides(intArrayOf(batchSize, seqLen))).asTensor("attention_mask"))
        input.add(LongNDArray(inputIds.flatten().indices.map { it.toLong() }.toLongArray(), Strides(intArrayOf(batchSize, seqLen))).asTensor("position_ids"))

        val shape = intArrayOf(2, batchSize, numAttentionHeads, 0, hiddenSize / numAttentionHeads)
        for (i in 0 until numLayer) {
            val emptyPast = FloatNDArray(floatArrayOf(0.0f), Strides(shape))
            input.add(emptyPast.asTensor("past_$i"))
        }

        return process(input, batchSize, seqLen)
    }

    @ExperimentalUnsignedTypes
    override fun getLogProbs(inputIds: List<List<Int>>, past: List<MutableNDArray>): Pair<List<List<MutableList<Double>>>, List<MutableNDArray>> {
        val batchSize = inputIds.size
        if (batchSize == 0) {
            return Pair(emptyList(), emptyList())
        }

        val seqLen = inputIds[0].size
        val pastLength = past[0].shape[3]

        val input = ArrayList<Tensor>()
        input.add(LongNDArray(inputIds.flatten().map { it.toLong() }.toLongArray(), Strides(intArrayOf(batchSize, seqLen))).asTensor("input_ids"))
        input.add(FloatNDArray(List(batchSize * (pastLength + seqLen)) { 1.0F }.toFloatArray(), Strides(intArrayOf(batchSize, pastLength + seqLen))).asTensor("attention_mask"))
        val positions = inputIds.map { (pastLength until pastLength + seqLen).map { it.toLong() }.toList() }
        input.add(LongNDArray(positions.flatten().toLongArray(), Strides(intArrayOf(batchSize, seqLen))).asTensor("position_ids"))

        past.forEachIndexed { i, state -> input.add(state.asTensor("past_$i")) }
        // (2, 1, 4, 4, 64)

        return process(input, batchSize, seqLen)
    }

    private fun process(input: ArrayList<Tensor>, batchSize: Int, seqLen: Int): Pair<List<List<MutableList<Double>>>, List<MutableNDArray>> {
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

        return Pair(probs, output.subList(1, output.size).map { (it as Tensor).data.toMutable() })
    }
}
