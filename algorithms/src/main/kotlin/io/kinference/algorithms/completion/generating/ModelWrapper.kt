package io.kinference.algorithms.completion.generating

import io.kinference.algorithms.completion.ModelConfig
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.model.Model
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*

data class ModelOutput(val logProbs: Array<DoubleArray>, val pastStates: List<NDArray>)

data class ModelOutputSeq(val logProbs: List<Array<DoubleArray>> = emptyList(), val pastStates: List<NDArray> = emptyList()) {
    fun lastLogProbs(): ModelOutput {
        val lastProbs: Array<DoubleArray> = Array(logProbs.size) { logProbs[it].last() }
        return ModelOutput(lastProbs, pastStates)
    }
}

interface ModelWrapper {
    fun initLogProbs(inputIds: Array<IntArray>): ModelOutputSeq

    fun initLastLogProbs(inputIds: Array<IntArray>): ModelOutput {
        return initLogProbs(inputIds).lastLogProbs()
    }

    fun getLogProbs(inputIds: Array<IntArray>, past: List<NDArray>): ModelOutputSeq

    fun getLastLogProbs(inputIds: IntArray, past: List<NDArray>): ModelOutput {
        return getLogProbs(Array(inputIds.size) { intArrayOf(inputIds[it]) }, past).lastLogProbs()
    }
}

class GPT2ModelWrapper(config: ModelConfig) : ModelWrapper {
    val model = Model.load(config.modelPath)
    private val numAttentionHeads = config.numAttentionHeads
    private val hiddenSize = config.hiddenSize
    private val numLayer = config.numLayer
    private val vocabSize = config.vocabSize
//    distilgpt2_l3_h12_d256_int8

    override fun initLogProbs(inputIds: Array<IntArray>): ModelOutputSeq {
        val batchSize = inputIds.size
        if (batchSize == 0) {
            return ModelOutputSeq()
        }

        val seqLen = inputIds[0].size
        val input = ArrayList<Tensor>()
        val longIds = inputIds.toLongArray()
        val indices = longIds.indices.toLongArray()
        input.add(LongNDArray(shape = intArrayOf(batchSize, seqLen)) { longIds[it] }.asTensor("input_ids"))
        input.add(FloatNDArray(shape = intArrayOf(batchSize, seqLen)) { 1f }.asTensor("attention_mask"))
        input.add(LongNDArray(shape = intArrayOf(batchSize, seqLen)) { indices[it] }.asTensor("position_ids"))

        val shape = intArrayOf(2, batchSize, numAttentionHeads, 0, hiddenSize / numAttentionHeads)
        for (i in 0 until numLayer) {
            val emptyPast = FloatNDArray(shape)
            input.add(emptyPast.asTensor("past_$i"))
        }

        return process(input, batchSize, seqLen)
    }

    override fun getLogProbs(inputIds: Array<IntArray>, past: List<NDArray>): ModelOutputSeq {
        val batchSize = inputIds.size
        if (batchSize == 0) {
            return ModelOutputSeq()
        }

        val seqLen = inputIds[0].size
        val pastLength = past[0].shape[3]

        val input = ArrayList<Tensor>()
        val longIds = inputIds.toLongArray()
        input.add(LongNDArray(intArrayOf(batchSize, seqLen)) { longIds[it] }.asTensor("input_ids"))
        input.add(FloatNDArray(shape = intArrayOf(batchSize, pastLength + seqLen)) { 1f }.asTensor("attention_mask"))
        val positions = LongArray(inputIds.size * seqLen).apply {
            for (i in 0 until seqLen) this[i] = (pastLength + i).toLong()
            for (i in 1 until inputIds.size) this.copyInto(this, i * seqLen, 0, seqLen)
        }
        input.add(LongNDArray(shape = intArrayOf(batchSize, seqLen)) { positions[it] }.asTensor("position_ids"))

        past.forEachIndexed { i, state -> input.add(state.asTensor("past_$i")) }
        // (2, 1, 4, 4, 64)

        return process(input, batchSize, seqLen)
    }

    private fun process(input: ArrayList<Tensor>, batchSize: Int, seqLen: Int): ModelOutputSeq {
        val output = model.predict(input).map { (it as Tensor).data }
        val ndProbs = (output[0] as FloatNDArray).array

        val probs = List(batchSize) { Array(seqLen) { DoubleArray(vocabSize) } }
        for (batch in 0 until batchSize) {
            val batchOff = batch * seqLen * vocabSize
            for (pos in 0 until seqLen) {
                val posOff = batchOff + pos * vocabSize
                for (id in 0 until vocabSize) {
                    probs[batch][pos][id] = (ndProbs[posOff + id]).toDouble()
                }
            }
        }

        return ModelOutputSeq(probs, output.drop(1))
    }
}
