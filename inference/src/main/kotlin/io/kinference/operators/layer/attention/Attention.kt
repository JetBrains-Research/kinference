package io.kinference.operators.layer.attention

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.accept
import io.kinference.ndarray.arrays.pointers.map
import io.kinference.ndarray.extensions.*
import io.kinference.onnx.AttributeProto
import io.kinference.onnx.TensorProto
import io.kinference.operators.*
import io.kinference.operators.activations.Softmax
import io.kinference.primitives.types.DataType
import kotlin.math.min
import kotlin.math.sqrt

class Attention(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {

    companion object {
        private val TYPE_CONSTRAINTS = setOf(TensorProto.DataType.FLOAT, TensorProto.DataType.FLOAT16)

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("num_heads", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("unidirectional", setOf(AttributeProto.AttributeType.INT), false, default = 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "weight", optional = false, divider = 3),
            IOInfo(2, TYPE_CONSTRAINTS, "bias", optional = false, divider = 3),
            IOInfo(3, setOf(TensorProto.DataType.INT32), "mask_index", optional = true),
            IOInfo(4, TYPE_CONSTRAINTS, "past", optional = true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "present", optional = true)
        )

        private val INFO = OperatorInfo("Attention", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)

        internal fun initQueryKeyValue(
            input: NDArray, weights: NDArray, bias: NDArray, batchSize: Int, seqLen: Int,
            hiddenSize: Int, numHeads: Int
        ): Array<MutableNDArray> {
            input as NumberNDArray
            val headSize = hiddenSize / numHeads
            val qkvWeights = weights.splitHorizontalByBlocks(3).map { it.splitHorizontalByBlocks(numHeads) }
            val qkvBias = bias.splitHorizontalByBlocks(3).map { it.splitHorizontalByBlocks(numHeads) }

            val qkv = Array(3) { allocateNDArray(input.type, Strides(intArrayOf(batchSize, numHeads, seqLen, headSize))) }

            for (qkvIdx in 0 until 3) {
                val output = qkv[qkvIdx]
                val weights = qkvWeights[qkvIdx]
                val bias = qkvBias[qkvIdx]

                for (batchNum in 0 until batchSize) {
                    val inputMatrix = input.view(batchNum)
                    for (numHead in 0 until numHeads) {
                        val outputMatrix = output.viewMutable(batchNum, numHead)
                        val weightsMatrix = weights[numHead]
                        val biasMatrix = bias[numHead]

                        (inputMatrix as NumberNDArray).dot(weightsMatrix as NumberNDArray, outputMatrix as MutableNumberNDArray)
                        outputMatrix.plusAssign(biasMatrix)
                    }
                }
            }

            return qkv
        }

        private fun IntNDArray?.maskFromIndices(unidir: Boolean, batchSize: Int, seqLen: Int, pastSeqLen: Int): FloatNDArray {
            val fullSeqLen = seqLen + pastSeqLen
            val maskDataShape = intArrayOf(batchSize, seqLen, fullSeqLen)
            val mask = allocateNDArray(DataType.FLOAT, Strides(maskDataShape)) as MutableFloatNDArray
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

        //create present state block from past + current states
        private fun MutableNDArray.updateState(past: NDArray?, currentState: NDArray, pastBlockSize: Int, presentBlockSize: Int, i: Int,
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

        private fun concatStateChunk(past: NDArray?, chunk: NDArray, present: MutableNDArray) {
            //TODO: Check iterators in copyFrom
            var additionalForChunkOffset = 0
            if (past != null) {
                present.copyFrom(0, past)
                additionalForChunkOffset += past.linearSize
            }
            present.copyFrom(additionalForChunkOffset, chunk)
        }

        private fun normalizedScores(
            unidir: Boolean, queries: NDArray, keys: NDArray, maskIndices: IntNDArray?, batchSize: Int,
            seqLen: Int, pastSeqLen: Int, headSize: Int, numHeads: Int, past: NDArray?, present: MutableNDArray
        ): NDArray {
            val allSeqLen = pastSeqLen + seqLen

            val scores = allocateNDArray(queries.type, Strides(intArrayOf(batchSize, numHeads, seqLen, allSeqLen)))

            val maskData = maskIndices?.maskFromIndices(unidir, batchSize, seqLen, pastSeqLen)

            val alpha = 1.0 / sqrt(headSize.toDouble())

            for (batchNum in 0 until batchSize) {
                for (numHead in 0 until numHeads) {
                    val queryMatrix = queries.view(batchNum, numHead)
                    val keyMatrix = keys.view(batchNum, numHead)
                    val pastMatrix = past?.view(0, batchNum, numHead)
                    val presentMatrix = present.viewMutable(0, batchNum, numHead)
                    val scoresMatrix = scores.viewMutable(batchNum, numHead)
                    val maskVector = maskData?.view(batchNum)

                    concatStateChunk(pastMatrix, keyMatrix, presentMatrix)
                    gemm(seqLen, allSeqLen, headSize, alpha, queryMatrix as NumberNDArray, presentMatrix as NumberNDArray, 1.0, scoresMatrix, transposeB = true)
                    if (maskVector != null)
                        (scoresMatrix as MutableNumberNDArray).plusAssign(maskVector)
                }
            }

            //softmax for each result (normalize along last axis)
            return Softmax.softmax(scores, -1)
        }

        private fun attentionScore(
            scores: NDArray, values: NDArray, batchSize: Int, seqLen: Int, pastSeqLen: Int,
            numHeads: Int, hiddenSize: Int, past: NDArray?, present: MutableNDArray
        ): Pair<NDArray, NDArray> {
            val headSize = hiddenSize / numHeads

            val output = allocateNDArray(scores.type, Strides(intArrayOf(batchSize, numHeads, seqLen, headSize)))

            for (batchNum in 0 until batchSize) {
                for (numHead in 0 until numHeads) {
                    val tempScores = scores.view(batchNum, numHead)
                    val tempOutput = output.viewMutable(batchNum, numHead)

                    val tempValues = values.view(batchNum, numHead)
                    val tempPast = past?.view(1, batchNum, numHead)
                    val tempPresent = present.viewMutable(1, batchNum, numHead)

                    concatStateChunk(tempPast, tempValues, tempPresent)
                    (tempScores as NumberNDArray).dot(tempPresent as NumberNDArray, tempOutput as MutableNumberNDArray)
                }
            }

            output.transpose(intArrayOf(0, 2, 1, 3))
            return output.reshapeView(intArrayOf(batchSize, seqLen, hiddenSize)) to present
        }

        internal fun getScores(
            unidir: Boolean, q: NDArray, k: NDArray, v: NDArray, mask: IntNDArray?,
            past: NDArray?, batchSize: Int, seqLen: Int, numHeads: Int, hiddenSize: Int
        ): Pair<NDArray, NDArray> {
            var pastSeqLen = 0
            val headSize = hiddenSize / numHeads
            val presentDims = intArrayOf(2, batchSize, numHeads, seqLen, headSize)
            if (past != null) {
                pastSeqLen = past.shape[3]
                presentDims[3] += pastSeqLen
            }

            val present = allocateNDArray(q.type, Strides(presentDims))

            val scores = normalizedScores(unidir, q, k, mask, batchSize, seqLen, pastSeqLen, headSize, numHeads, past, present)
            return attentionScore(scores, v, batchSize, seqLen, pastSeqLen, numHeads, hiddenSize, past, present)
        }
    }

    private val numHeads: Int by attribute("num_heads") { it: Number -> it.toInt() }
    private val unidir: Boolean by attribute("unidirectional") { it: Number -> it.toInt() == 1 }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val input = inputs[0]!!.data
        val weights = inputs[1]!!.data
        val bias = inputs[2]!!.data
        val maskIndices = inputs.elementAtOrNull(3)?.data as IntNDArray?
        val past = inputs.elementAtOrNull(4)?.data

        val (batchSize, seqLen, hiddenSize) = input.shape

        val (queries, keys, values) = initQueryKeyValue(input, weights, bias, batchSize, seqLen, hiddenSize, numHeads)

        val (scores, present) = getScores(unidir, queries, keys, values, maskIndices, past, batchSize, seqLen, numHeads, hiddenSize)
        return listOf(scores.asTensor(), present.asTensor())
    }
}
