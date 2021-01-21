package io.kinference.algorithms.completion.generation.model

import io.kinference.algorithms.completion.CompletionConfig
import io.kinference.algorithms.completion.generation.toLongArray
import io.kinference.algorithms.completion.loader.CompletionModelLoader
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.model.Model
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.LongTiledArray

internal class GPT2ModelWrapper(loader: CompletionModelLoader, config: CompletionConfig.Model) : ModelWrapper {
    private val model = Model.load(loader.getModel())
    private val numAttentionHeads = config.numAttentionHeads
    private val hiddenSize = config.hiddenSize
    private val numLayer = config.numLayer
    private val vocabSize = config.vocabSize
    override val maxSeqLen = config.maxSeqLen

    override fun initLogProbs(inputIds: Array<IntArray>): ModelOutputSeq {
        val batchSize = inputIds.size
        if (batchSize == 0) {
            return ModelOutputSeq()
        }

        val seqLen = inputIds[0].size
        val input = ArrayList<Tensor>()
        val longIds = LongTiledArray(Array(inputIds.size) { inputIds.toLongArray() })
        input.add(LongNDArray(shape = intArrayOf(batchSize, seqLen)) { longIds[it] }.asTensor("input_ids"))
        input.add(FloatNDArray(shape = intArrayOf(batchSize, seqLen)) { 1f }.asTensor("attention_mask"))
        input.add(LongNDArray(shape = intArrayOf(batchSize, seqLen)) { it.toLong() }.asTensor("position_ids"))

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
        val longIds = LongTiledArray(Array(inputIds.size) { inputIds[it].toLongArray() })
        input.add(LongNDArray(longIds, Strides(intArrayOf(batchSize, seqLen))).asTensor("input_ids"))
        input.add(FloatNDArray(shape = intArrayOf(batchSize, pastLength + seqLen)) { 1f }.asTensor("attention_mask"))
        input.add(LongNDArray(shape = intArrayOf(batchSize, seqLen)) { (pastLength + it % seqLen).toLong() }.asTensor("position_ids"))

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
