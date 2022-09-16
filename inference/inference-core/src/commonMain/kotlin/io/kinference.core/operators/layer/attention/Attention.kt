package io.kinference.core.operators.layer.attention

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.accept
import io.kinference.ndarray.arrays.pointers.map
import io.kinference.operator.*
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.graph.asCoroutineContext
import io.kinference.model.ExecutionContext
import io.kinference.ndarray.extensions.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.runBlocking
import kotlinx.coroutines.*
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.time.ExperimentalTime

sealed class Attention(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private fun attentionScore(
            scores: NDArrayCore, values: NDArrayCore, batchSize: Int, seqLen: Int, pastSeqLen: Int,
            numHeads: Int, hiddenSize: Int, past: NDArrayCore?, present: MutableNDArrayCore,
            executionContext: ExecutionContext?
        ): Pair<NDArrayCore, NDArrayCore> {
            val headSize = hiddenSize / numHeads

            val output = allocateNDArray(scores.type, Strides(intArrayOf(batchSize, numHeads, seqLen, headSize)))

            runBlocking(executionContext.asCoroutineContext()) {
                for (batchNum in 0 until batchSize) {
                    for (numHead in 0 until numHeads) {
                        launch {
                            val tempScores = scores.view(batchNum, numHead) as NumberNDArrayCore
                            val tempOutput = output.viewMutable(batchNum, numHead) as MutableNumberNDArray

                            val tempValues = values.view(batchNum, numHead)
                            val tempPast = past?.view(1, batchNum, numHead)
                            val tempPresent = present.viewMutable(1, batchNum, numHead)

                            concatStateChunk(tempPast, tempValues, tempPresent)
                            tempScores.dot(tempPresent as NumberNDArray, tempOutput, executionContext.asCoroutineContext())
                        }
                    }
                }
            }

            val outputTransposed = output.transpose(intArrayOf(0, 2, 1, 3)).reshape(intArrayOf(batchSize, seqLen, hiddenSize))
            return outputTransposed to present
        }

        internal fun getScores(
            unidir: Boolean, q: NDArrayCore, k: NDArrayCore, v: NDArrayCore, mask: IntNDArray?,
            past: NDArrayCore?, batchSize: Int, seqLen: Int, numHeads: Int, hiddenSize: Int,
            executionContext: ExecutionContext?
        ): Pair<NDArrayCore, NDArrayCore> {
            var pastSeqLen = 0
            val headSize = hiddenSize / numHeads
            val presentDims = intArrayOf(2, batchSize, numHeads, seqLen, headSize)
            if (past != null) {
                pastSeqLen = past.shape[3]
                presentDims[3] += pastSeqLen
            }

            val present = allocateNDArray(q.type, Strides(presentDims))

            val scores = normalizedScores(unidir, q, k, mask, batchSize, seqLen, pastSeqLen, headSize, numHeads, past, present, executionContext)
            return attentionScore(scores, v, batchSize, seqLen, pastSeqLen, numHeads, hiddenSize, past, present, executionContext)
        }

        private fun normalizedScores(
            unidir: Boolean, queries: NDArrayCore, keys: NDArrayCore, maskIndices: IntNDArray?, batchSize: Int,
            seqLen: Int, pastSeqLen: Int, headSize: Int, numHeads: Int, past: NDArrayCore?, present: MutableNDArrayCore,
            executionContext: ExecutionContext?
        ): NumberNDArrayCore {
            val allSeqLen = pastSeqLen + seqLen

            val scores = allocateNDArray(queries.type, Strides(intArrayOf(batchSize, numHeads, seqLen, allSeqLen))) as MutableNumberNDArrayCore

            val maskData = maskIndices?.maskFromIndices(unidir, batchSize, seqLen, pastSeqLen)

            val alpha = 1.0 / sqrt(headSize.toDouble())

            runBlocking(executionContext.asCoroutineContext()) {
                for (batchNum in 0 until batchSize) {
                    for (numHead in 0 until numHeads) {
                        launch {
                            val queryMatrix = queries.view(batchNum, numHead)
                            val keyMatrix = keys.view(batchNum, numHead)
                            val pastMatrix = past?.view(0, batchNum, numHead)
                            val presentMatrix = present.viewMutable(0, batchNum, numHead) as MutableNumberNDArrayCore
                            val scoresMatrix = scores.viewMutable(batchNum, numHead) as MutableNumberNDArray
                            val maskVector = maskData?.view(batchNum)

                            concatStateChunk(pastMatrix, keyMatrix, presentMatrix)
                            (queryMatrix as FloatNDArray).dotTransposedWithAlpha(alpha, presentMatrix, scoresMatrix, executionContext.asCoroutineContext())
//                    gemm(seqLen, allSeqLen, headSize, alpha, queryMatrix as NumberNDArray, presentMatrix as NumberNDArray, 1.0, scoresMatrix, transposeB = true)
                            if (maskVector != null)
                                scoresMatrix.plusAssign(maskVector)
                        }
                    }
                }
            }

            //softmax for each result (normalize along last axis)
            return scores.softmax(axis = -1, coroutineContext = executionContext.asCoroutineContext())
        }

        private fun IntNDArray?.maskFromIndices(unidir: Boolean, batchSize: Int, seqLen: Int, pastSeqLen: Int): FloatNDArray {
            val fullSeqLen = seqLen + pastSeqLen
            val maskDataShape = intArrayOf(batchSize, seqLen, fullSeqLen)
            val mask = MutableFloatNDArray(Strides(maskDataShape))
            val maskOffset = seqLen * fullSeqLen
            repeat(batchSize) { i ->
                if (this != null) {
                    //raw attention (no padding). only raw attention mask is 2-dimensional
                    if (this.rank == 2) {
                        val maskPointer = mask.array.pointer(maskOffset * i)
                        val maskIndicesPointer = this.array.pointer(i * fullSeqLen)

                        maskPointer.accept(maskIndicesPointer, fullSeqLen) { _, src -> if (src > 0) 0f else -10000f }
                    } else {
                        //for left/right-side padding
                        val maskIndicesPointer = this.array.pointer(i)
                        val maskPointer = mask.array.pointer(maskOffset * i + maskIndicesPointer.get())
                        maskPointer.map(fullSeqLen - maskIndicesPointer.get()) { -10000f }

                        if (this.rank == 1 && this.shape[0] == 2 * batchSize) {
                            maskIndicesPointer.linearIndex = i + batchSize
                            maskPointer.linearIndex = maskOffset * i
                            maskPointer.map(min(maskIndicesPointer.get(), fullSeqLen)) { -10000f }
                        }
                    }
                }

                //broadcast mask block
                for (seqIdx in 1 until seqLen) {
                    val start = seqIdx * fullSeqLen + i * maskOffset
                    mask.copyFrom(start, mask, i * maskOffset, i * maskOffset + fullSeqLen)
                }

                if (unidir) {
                    val maskPointer = mask.array.pointer()
                    for (seqIdx in 0 until seqLen - 1) {
                        val start = pastSeqLen + seqIdx + 1
                        maskPointer.linearIndex = seqIdx * fullSeqLen + maskOffset * i + start
                        maskPointer.map(fullSeqLen - start) { it - 10000f }
                    }
                }
            }
            return mask
        }

        private fun concatStateChunk(past: NDArrayCore?, chunk: NDArrayCore, present: MutableNDArrayCore) {
            //TODO: Check iterators in copyFrom
            var additionalForChunkOffset = 0
            if (past != null) {
                present.copyFrom(0, past)
                additionalForChunkOffset += past.linearSize
            }
            present.copyFrom(additionalForChunkOffset, chunk)
        }

        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in AttentionVer1.VERSION.asRange() -> AttentionVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Attention operator: $version")
        }
    }
}

@ExperimentalTime
class AttentionVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Attention(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(TensorProto.DataType.FLOAT, TensorProto.DataType.FLOAT16)

        val ATTRIBUTES_INFO = listOf(
            AttributeInfo("num_heads", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("unidirectional", setOf(AttributeProto.AttributeType.INT), false, default = 0)
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
        val INFO = OperatorInfo("Attention", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")

        internal fun initQueryKeyValue(
            input: NDArrayCore, weights: NDArrayCore, bias: NDArrayCore, batchSize: Int,
            seqLen: Int, hiddenSize: Int, numHeads: Int, executionContext: ExecutionContext?
        ): Array<MutableNDArrayCore> {
            input as NumberNDArrayCore
            val headSize = hiddenSize / numHeads

            val qkv = Array(3) { allocateNDArray(input.type, Strides(intArrayOf(batchSize, numHeads, seqLen, headSize))) }

            runBlocking(executionContext.asCoroutineContext()) {
                for (qkvIdx in 0 until 3) {
                    launch {
                        val output = qkv[qkvIdx]
                        for (batchNum in 0 until batchSize) {
                            val inputMatrix = input.view(batchNum)
                            for (numHead in 0 until numHeads) {
                                val weightsMatrix = weights.view(qkvIdx, numHead) as NumberNDArrayCore
                                val biasMatrix = bias.view(qkvIdx, numHead) as NumberNDArray

                                val outputMatrix = output.viewMutable(batchNum, numHead)

                                inputMatrix.dot(weightsMatrix, outputMatrix as MutableNumberNDArray, executionContext.asCoroutineContext())
                                outputMatrix.plusAssign(biasMatrix)
                            }
                        }
                    }
                }
            }

            return qkv
        }

        //create present state block from past + current states
        private fun MutableNDArrayCore.updateState(past: NDArray?, currentState: NDArray, pastBlockSize: Int, presentBlockSize: Int, i: Int,
                                               pastOffset: Int, presentOffset: Int, currentOffset: Int): Pair<MutableNDArray, Int> {
            //present state block offset
            val presentStart = i * presentBlockSize + presentOffset

            var presentPos = presentStart
            if (past != null) {
                val srcPast = i * pastBlockSize + pastOffset
                this.copyFrom(presentPos, past, srcPast, srcPast + pastBlockSize)
                presentPos += pastBlockSize
            }
            this.copyFrom(presentPos, currentState, currentOffset, currentOffset + presentBlockSize - pastBlockSize)

            return this to presentStart
        }
    }

    private val numHeads: Int by attribute("num_heads") { it: Number -> it.toInt() }
    private val unidir: Boolean by attribute("unidirectional") { it: Number -> it.toInt() == 1 }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!
        val weights = inputs[1]!!
        val preparedWeights = (contexts.graph!!.getOrNullValue("prepared_${weights.name}") ?: AttentionContext.prepareWeights(weights, numHeads)) as KITensor

        val bias = inputs[2]!!
        val preparedBias = (contexts.graph!!.getOrNullValue("prepared_${bias.name}") ?: AttentionContext.prepareBias(bias, numHeads)) as KITensor

        val maskIndices = inputs.elementAtOrNull(3)?.data as IntNDArray?
        val past = inputs.elementAtOrNull(4)?.data

        val (batchSize, seqLen, hiddenSize) = input.data.shape

        val (queries, keys, values) = initQueryKeyValue(
            input.data,
            preparedWeights.data,
            preparedBias.data,
            batchSize, seqLen, hiddenSize, numHeads,
            contexts.execution
        )

        val (scores, present) = getScores(unidir, queries, keys, values, maskIndices, past, batchSize, seqLen, numHeads, hiddenSize, contexts.execution)
        return listOf(scores.asTensor(), present.asTensor())
    }
}
