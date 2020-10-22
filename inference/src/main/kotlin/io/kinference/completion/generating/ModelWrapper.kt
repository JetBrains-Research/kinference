package io.kinference.completion.generating

import io.kinference.completion.ModelConfig
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.model.Model
import io.kinference.ndarray.*

interface ModelWrapper {
    fun initLogProbs(inputIds: Array<IntArray>): Pair<List<Array<DoubleArray>>, List<MutableNDArray>>

    fun initLastLogProbs(inputIds: Array<IntArray>): Pair<Array<DoubleArray>, List<MutableNDArray>> {
        val (score, mems) = initLogProbs(inputIds)
        // (batchSize, seqLen, vocabSize) -> (batchSize, vocabSize)
        val lastProbs: Array<DoubleArray> = Array(score.size) { score[it].last() }
        return Pair(lastProbs, mems)
    }

    fun getLogProbs(inputIds: Array<IntArray>, past: List<MutableNDArray>): Pair<List<Array<DoubleArray>>, List<MutableNDArray>>

    fun getLastLogProbs(inputIds: IntArray, past: List<MutableNDArray>): Pair<Array<DoubleArray>, List<MutableNDArray>> {
        val (score, mems) = getLogProbs(Array(inputIds.size) { intArrayOf(inputIds[it]) }, past)
        // (batchSize, seqLen, vocabSize) -> (batchSize, vocabSize)
        val lastProbs: Array<DoubleArray> = Array(score.size) { score[it].last() }
        return Pair(lastProbs, mems)
    }
}

class OnnxModelWrapper(config: ModelConfig) : ModelWrapper {
    val model = Model.load(config.modelPath)
    private val numAttentionHeads = config.numAttentionHeads
    private val hiddenSize = config.hiddenSize
    private val numLayer = config.numLayer
    private val vocabSize = config.vocabSize
//    distilgpt2_l3_h12_d256_int8

    @ExperimentalUnsignedTypes
    override fun initLogProbs(inputIds: Array<IntArray>): Pair<List<Array<DoubleArray>>, List<MutableNDArray>> {
        val batchSize = inputIds.size
        if (batchSize == 0) {
            return Pair(emptyList(), emptyList())
        }

        val seqLen = inputIds[0].size
        val input = ArrayList<Tensor>()
        val longIds = inputIds.toLongArray()
        input.add(LongNDArray(longIds, Strides(intArrayOf(batchSize, seqLen))).asTensor("input_ids"))
        input.add(FloatNDArray(FloatArray(batchSize * seqLen) { 1f }, Strides(intArrayOf(batchSize, seqLen))).asTensor("attention_mask"))
        input.add(LongNDArray(longIds.indices.toLongArray(), Strides(intArrayOf(batchSize, seqLen))).asTensor("position_ids"))

        val shape = intArrayOf(2, batchSize, numAttentionHeads, 0, hiddenSize / numAttentionHeads)
        for (i in 0 until numLayer) {
            val emptyPast = FloatNDArray(FloatArray(0), Strides(shape))
            input.add(emptyPast.asTensor("past_$i"))
        }

        return process(input, batchSize, seqLen)
    }

    @ExperimentalUnsignedTypes
    override fun getLogProbs(inputIds: Array<IntArray>, past: List<MutableNDArray>): Pair<List<Array<DoubleArray>>, List<MutableNDArray>> {
        val batchSize = inputIds.size
        if (batchSize == 0) {
            return Pair(emptyList(), emptyList())
        }

        val seqLen = inputIds[0].size
        val pastLength = past[0].shape[3]

        val input = ArrayList<Tensor>()
        input.add(LongNDArray(inputIds.toLongArray(), Strides(intArrayOf(batchSize, seqLen))).asTensor("input_ids"))
        input.add(FloatNDArray(FloatArray(batchSize * (pastLength + seqLen)) { 1f }, Strides(intArrayOf(batchSize, pastLength + seqLen))).asTensor("attention_mask"))
        val positions = LongArray(inputIds.size * seqLen).apply {
            for (i in 0 until seqLen) this[i] = (pastLength + i).toLong()
            for (i in 1 until inputIds.size) this.copyInto(this, i * seqLen, 0, seqLen)
        }
        input.add(LongNDArray(positions, Strides(intArrayOf(batchSize, seqLen))).asTensor("position_ids"))

        past.forEachIndexed { i, state -> input.add(state.asTensor("past_$i")) }
        // (2, 1, 4, 4, 64)

        return process(input, batchSize, seqLen)
    }

    private fun process(input: ArrayList<Tensor>, batchSize: Int, seqLen: Int): Pair<List<Array<DoubleArray>>, List<MutableNDArray>> {
        val output = model.predict(input)
        val ndProbs = (output[0] as Tensor).data

        val probs: List<Array<DoubleArray>> = List(batchSize) { Array(seqLen) { DoubleArray(vocabSize) } }
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
