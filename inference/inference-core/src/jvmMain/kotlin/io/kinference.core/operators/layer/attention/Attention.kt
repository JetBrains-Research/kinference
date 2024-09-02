package io.kinference.core.operators.layer.attention

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.optimizer.rules.context.AttentionContextRule
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.memory.contexts.ManualAllocatorContext
import io.kinference.ndarray.arrays.pointers.accept
import io.kinference.ndarray.arrays.pointers.map
import io.kinference.ndarray.arrays.tiled.FloatTiledArray
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.extensions.dotTransposedWithAlpha
import io.kinference.ndarray.extensions.softmax.softmax
import io.kinference.operator.*
import io.kinference.optimizer.GraphOptimizer.Companion.isOpt
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.launchWithLimitOrDefault
import kotlinx.coroutines.coroutineScope
import kotlin.coroutines.coroutineContext
import kotlin.math.min
import kotlin.math.sqrt

sealed class Attention(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private suspend fun attentionScore(
            scores: NDArrayCore, batchSize: Int, seqLen: Int,
            numHeads: Int, hiddenSize: Int, present: NDArrayCore, context: ManualAllocatorContext? = null
        ): Pair<NDArrayCore, NDArrayCore> {
            val headSize = hiddenSize / numHeads

            val outputStrides = Strides(intArrayOf(batchSize, numHeads, seqLen, headSize))
            val output = context?.getNDArray(scores.type, outputStrides, fillZeros = true) ?: allocateNDArray(scores.type, outputStrides)

            coroutineScope {
                for (batchNum in 0 until batchSize) {
                    for (numHead in 0 until numHeads) {
                        launchWithLimitOrDefault {
                            val tempScores = scores.view(batchNum, numHead) as NumberNDArrayCore
                            val tempOutput = output.viewMutable(batchNum, numHead) as MutableNumberNDArray

                            val tempPresent = present.view(1, batchNum, numHead)

                            tempScores.dot(tempPresent as NumberNDArray, tempOutput)
                        }
                    }
                }
            }

            context?.returnNDArray(scores)

            val outputTransposed = output.transpose(intArrayOf(0, 2, 1, 3)).reshape(intArrayOf(batchSize, seqLen, hiddenSize))
            return outputTransposed to present
        }

        private fun makePresent(past: NDArrayCore?, k: NDArrayCore, v: NDArrayCore, batchSize: Int, seqLen: Int, numHeads: Int, hiddenSize: Int): FloatNDArray {
            past as FloatNDArray?
            k as FloatNDArray
            v as FloatNDArray

            val headSize = hiddenSize / numHeads
            val presentDims = intArrayOf(2, batchSize, numHeads, seqLen, headSize)

            val kBlocks = k.array.blocks
            val vBlocks = v.array.blocks


            val resultBlocks: Array<FloatArray>

            if (past == null || past.linearSize == 0) {
                resultBlocks = kBlocks.plus(vBlocks)
            } else {
                val pastSeqLen = past.shape[3]
                presentDims[3] += pastSeqLen

                val pastBlocks = past.array.blocks

                val blocksInRow = headSize / past.array.blockSize

                val pastRowBlocksCount = pastSeqLen * blocksInRow
                val kvRowBlocksCount = seqLen * blocksInRow

                val rowsSize = batchSize * numHeads
                val futureRes = arrayOfNulls<FloatArray>(2 * batchSize * numHeads * presentDims[3] * blocksInRow)

                var resBlockIdx = 0
                var pastBlocIdx = 0

                repeat(2) { presentKeyValueIdx ->
                    val kvBlocks = if (presentKeyValueIdx == 0) kBlocks else vBlocks

                    var kvBlockIdx = 0

                    repeat(rowsSize) {
                        pastBlocks.copyInto(futureRes, resBlockIdx, pastBlocIdx, pastBlocIdx + pastRowBlocksCount)

                        resBlockIdx += pastRowBlocksCount
                        pastBlocIdx += pastRowBlocksCount

                        kvBlocks.copyInto(futureRes, resBlockIdx, kvBlockIdx, kvBlockIdx + kvRowBlocksCount)
                        resBlockIdx += kvRowBlocksCount
                        kvBlockIdx += kvRowBlocksCount
                    }
                }
                resultBlocks = futureRes as Array<FloatArray>
            }

            return FloatNDArray(FloatTiledArray(resultBlocks), Strides(presentDims))
        }


        internal suspend fun getScores(
            unidir: Boolean, q: NDArrayCore, k: NDArrayCore, v: NDArrayCore, mask: IntNDArray?,
            past: NDArrayCore?, batchSize: Int, seqLen: Int, numHeads: Int, hiddenSize: Int, maskFilterValue: Float = -10_000f, context: ManualAllocatorContext? = null
        ): Pair<NDArrayCore, NDArrayCore> {
            val headSize = hiddenSize / numHeads

            val pastSeqLen = past?.shape?.get(3) ?: 0
            val present = makePresent(past, k, v, batchSize, seqLen, numHeads, hiddenSize)

            val scores = normalizedScores(unidir, q, mask, batchSize, seqLen, pastSeqLen, headSize, numHeads, present, maskFilterValue, context)
            return attentionScore(scores, batchSize, seqLen, numHeads, hiddenSize, present, context)
        }

        private suspend fun normalizedScores(
            unidir: Boolean, queries: NDArrayCore, maskIndices: IntNDArray?, batchSize: Int,
            seqLen: Int, pastSeqLen: Int, headSize: Int, numHeads: Int, present: NDArrayCore, maskFilterValue: Float = -10_000f, context: ManualAllocatorContext? = null
        ): NumberNDArrayCore {
            val allSeqLen = present.shape[3]

            val scoresStrides = Strides(intArrayOf(batchSize, numHeads, seqLen, allSeqLen))
            val scores = (context?.getNDArray(queries.type, scoresStrides, fillZeros = true) ?: allocateNDArray(queries.type, scoresStrides)) as MutableNumberNDArrayCore

            val maskData = maskIndices?.maskFromIndices(unidir, batchSize, seqLen, pastSeqLen, maskFilterValue, context)

            val alpha = 1.0 / sqrt(headSize.toDouble())

            coroutineScope {
                for (batchNum in 0 until batchSize) {
                    for (numHead in 0 until numHeads) {
                        launchWithLimitOrDefault {
                            val queryMatrix = queries.view(batchNum, numHead)
                            val presentMatrix = present.view(0, batchNum, numHead) as NumberNDArray
                            val scoresMatrix = scores.viewMutable(batchNum, numHead) as MutableNumberNDArray
                            val maskVector = maskData?.view(batchNum)

                            (queryMatrix as FloatNDArray).dotTransposedWithAlpha(alpha, presentMatrix, scoresMatrix)
                            if (maskVector != null)
                                scoresMatrix.plusAssign(maskVector)
                        }
                    }
                }
            }

            if (maskData != null) {
                context?.returnNDArray(maskData)
            }
            context?.returnNDArray(queries)

            val softmaxDest = (context?.getNDArray(scores.type, scoresStrides) ?: allocateNDArray(scores.type, scoresStrides)) as MutableNumberNDArrayCore

            //softmax for each result (normalize along last axis)
            return softmax(input = scores, axis = -1, dest = softmaxDest)
        }

        private suspend fun IntNDArray?.maskFromIndices(unidir: Boolean, batchSize: Int, seqLen: Int, pastSeqLen: Int, maskFilterValue: Float = -10_000f, context: ManualAllocatorContext? = null): FloatNDArray {
            val fullSeqLen = seqLen + pastSeqLen
            val maskDataShape = intArrayOf(batchSize, seqLen, fullSeqLen)
            val maskStrides = Strides(maskDataShape)

            val mask = (context?.getNDArray(DataType.FLOAT, maskStrides) ?: MutableFloatNDArray(maskStrides)) as MutableFloatNDArray
            val maskOffset = seqLen * fullSeqLen
            repeat(batchSize) { i ->
                if (this != null) {
                    //raw attention (no padding). only raw attention mask is 2-dimensional
                    if (this.rank == 2) {
                        val maskPointer = mask.array.pointer(maskOffset * i)
                        val maskIndicesPointer = this.array.pointer(i * fullSeqLen)

                        maskPointer.accept(maskIndicesPointer, fullSeqLen) { _, src -> if (src > 0) 0f else maskFilterValue }
                    } else {
                        //for left/right-side padding
                        val maskIndicesPointer = this.array.pointer(i)
                        val maskPointer = mask.array.pointer(maskOffset * i + maskIndicesPointer.get())
                        maskPointer.map(fullSeqLen - maskIndicesPointer.get()) { maskFilterValue }

                        if (this.rank == 1 && this.shape[0] == 2 * batchSize) {
                            maskIndicesPointer.linearIndex = i + batchSize
                            maskPointer.linearIndex = maskOffset * i
                            maskPointer.map(min(maskIndicesPointer.get(), fullSeqLen)) { maskFilterValue }
                        }
                    }

                    //broadcast mask block
                    for (seqIdx in 1 until seqLen) {
                        val start = seqIdx * fullSeqLen + i * maskOffset
                        mask.copyFrom(start, mask, i * maskOffset, i * maskOffset + fullSeqLen)
                    }
                }

                if (unidir) {
                    val maskPointer = mask.array.pointer()
                    for (seqIdx in 0 until seqLen - 1) {
                        val start = pastSeqLen + seqIdx + 1
                        maskPointer.linearIndex = seqIdx * fullSeqLen + maskOffset * i + start
                        maskPointer.map(fullSeqLen - start) { it + maskFilterValue }
                    }
                }
            }
            return mask
        }

        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in AttentionVer1.VERSION.asRange() -> AttentionVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Attention operator: $version")
        }
    }
}


class AttentionVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Attention(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(TensorProto.DataType.FLOAT, TensorProto.DataType.FLOAT16)

        val ATTRIBUTES_INFO = listOf(
            AttributeInfo("num_heads", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("unidirectional", setOf(AttributeProto.AttributeType.INT), false, default = 0),
            AttributeInfo("mask_filter_value", setOf(AttributeProto.AttributeType.FLOAT), false, default = -10_000f)
        )

        val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "weight", optional = false),
            IOInfo(2, TYPE_CONSTRAINTS, "bias", optional = false),
            IOInfo(3, setOf(TensorProto.DataType.INT32), "mask_index", optional = true),
            IOInfo(4, TYPE_CONSTRAINTS, "past", optional = true)
        )

        val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "present", optional = true)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        val INFO = OperatorInfo("Attention", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.ORT_DOMAIN)

        internal suspend fun initQueryKeyValue(
            input: NDArrayCore, weights: NDArrayCore, bias: NDArrayCore,
            batchSize: Int, seqLen: Int, hiddenSize: Int, numHeads: Int, context: ManualAllocatorContext? = null
        ): Array<MutableNDArrayCore> {
            input as NumberNDArrayCore
            val headSize = hiddenSize / numHeads

            val qkvStrides = Strides(intArrayOf(batchSize, numHeads, seqLen, headSize))
            val qkv = Array(3) { context?.getNDArray(input.type, qkvStrides, fillZeros = true) ?: allocateNDArray(input.type, qkvStrides) }

            coroutineScope {
                for (qkvIdx in 0 until 3) {
                    val output = qkv[qkvIdx]
                    for (batchNum in 0 until batchSize) {
                        val inputMatrix = input.view(batchNum)
                        for (numHead in 0 until numHeads) launchWithLimitOrDefault {
                            val weightsMatrix = weights.view(qkvIdx, numHead) as NumberNDArrayCore
                            val biasMatrix = bias.view(qkvIdx, numHead) as NumberNDArray

                            val outputMatrix = output.viewMutable(batchNum, numHead)

                            inputMatrix.dot(weightsMatrix, outputMatrix as MutableNumberNDArray)
                            outputMatrix.plusAssign(biasMatrix)
                        }
                    }
                }
            }

            return qkv
        }
    }

    private val numHeads: Int by attribute("num_heads") { it: Number -> it.toInt() }
    private val unidir: Boolean by attribute("unidirectional") { it: Number -> it.toInt() == 1 }
    private val maskFilterValue: Float by attribute("mask_filter_value") { it: Number -> it.toFloat() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val context = coroutineContext[ManualAllocatorContext.Key]

        val input = inputs[0]!!
        val weights = inputs[1]!!

        val preparedWeights = weights.takeIf { isOpt(it.name) } ?: AttentionContextRule.prepareWeights(weights, numHeads)

        val bias = inputs[2]!!
        val preparedBias = bias.takeIf { isOpt(it.name) } ?: AttentionContextRule.prepareBias(bias, numHeads)

        val maskIndices = inputs.elementAtOrNull(3)?.data as IntNDArray?
        val past = inputs.elementAtOrNull(4)?.data

        val (batchSize, seqLen, hiddenSize) = input.data.shape

        val (queries, keys, values) = initQueryKeyValue(
            input.data,
            preparedWeights.data,
            preparedBias.data,
            batchSize, seqLen, hiddenSize, numHeads, context
        )

        val (scores, present) = getScores(unidir, queries, keys, values, maskIndices, past, batchSize, seqLen, numHeads, hiddenSize, maskFilterValue, context)
        return listOf(scores.asTensor(context = context), present.asTensor(context = context))
    }
}
