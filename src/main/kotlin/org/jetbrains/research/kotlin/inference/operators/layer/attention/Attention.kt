package org.jetbrains.research.kotlin.inference.operators.layer.attention

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ndarray.FloatNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.MutableFloatNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.MutableTypedNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.TypedNDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.ndarray.allocateNDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.asTensor
import org.jetbrains.research.kotlin.inference.extensions.primitives.gemm
import org.jetbrains.research.kotlin.inference.graph.Context
import org.jetbrains.research.kotlin.inference.onnx.AttributeProto
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.AttributeInfo
import org.jetbrains.research.kotlin.inference.operators.IOInfo
import org.jetbrains.research.kotlin.inference.operators.Operator
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo
import org.jetbrains.research.kotlin.inference.operators.activations.Softmax
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
            IOInfo(1, TYPE_CONSTRAINTS, "weight", optional = false),
            IOInfo(2, TYPE_CONSTRAINTS, "bias", optional = false),
            IOInfo(3, setOf(TensorProto.DataType.INT32), "mask_index", optional = true),
            IOInfo(4, TYPE_CONSTRAINTS, "past", optional = true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "present", optional = true)
        )

        private val INFO = OperatorInfo("Attention", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)

        private fun initQueryKeyValue(input: TypedNDArray<Any>, weights: TypedNDArray<Any>, bias: TypedNDArray<Any>, batchSize: Int, seqLen: Int, hiddenSize: Int, numHeads: Int): Array<MutableTypedNDArray<Any>> {
            val qkv = Array(3) { allocateNDArray<Any>(input.type, Strides(intArrayOf(batchSize, seqLen, hiddenSize))) }
            val attentionHeadSize = hiddenSize / numHeads

            val step = batchSize * numHeads

            for (i in 0 until 3 * step) {
                val batchIdx = (i / 3) / numHeads
                val headIdx = (i / 3) % numHeads
                val qkvIdx = i % 3

                val inputBatchOffset = batchIdx * seqLen * hiddenSize
                val weightsOffset = qkvIdx * hiddenSize + headIdx * attentionHeadSize
                val qkvOffset = (batchIdx * numHeads + headIdx) * (seqLen * attentionHeadSize)

                //broadcast biases for each
                repeat(seqLen) {
                    val offset = qkvOffset + it * attentionHeadSize
                    qkv[qkvIdx].place(offset, bias.array, weightsOffset, weightsOffset + attentionHeadSize)
                }

                //x * W[q|k|v] + bias and apply mask
                gemm(input, weights, qkv[qkvIdx], inputBatchOffset, weightsOffset,
                    qkvOffset, seqLen, attentionHeadSize, hiddenSize,
                    hiddenSize, 3 * hiddenSize, attentionHeadSize)
            }
            return qkv
        }


        private fun TypedNDArray<Any>?.maskFromIndices(unidir: Boolean, batchSize: Int, seqLen: Int, pastSeqLen: Int): FloatNDArray {
            val fullSeqLen = seqLen + pastSeqLen
            val maskDataShape = intArrayOf(batchSize, seqLen, fullSeqLen)
            val mask = allocateNDArray<FloatArray>(TensorProto.DataType.FLOAT, Strides(maskDataShape)) as MutableFloatNDArray
            val maskOffset = seqLen * fullSeqLen
            repeat(batchSize) { i ->
                if (this != null) {
                    //raw attention (no padding). only raw attention mask is 2-dimensional
                    if (this.rank == 2) {
                        val indicesOffset = i * fullSeqLen
                        for (j in 0 until fullSeqLen) {
                            mask[maskOffset * i + j] = if ((this[j + indicesOffset] as Number).toInt() > 0) 0f else -10000f
                        }
                    } else {
                        //for left/right-side padding
                        val endPos = this[i] as Int
                        for (j in endPos until fullSeqLen) {
                            mask[i * maskOffset + j] = -10000f
                        }

                        if (this.rank == 1 && this.shape[0] == 2 * batchSize) {
                            val startPos = min((this[i + batchSize] as Number).toInt(), fullSeqLen)
                            for (j in 0 until startPos) {
                                mask[maskOffset * i + j] = -10000f
                            }
                        }
                    }
                }
                //broadcast mask block
                for (seqIdx in 1 until seqLen) {
                    val start = seqIdx * fullSeqLen + i * maskOffset
                    mask.place(start, mask.array, i * maskOffset, i * maskOffset + fullSeqLen)
                }

                if (unidir) {
                    for (seqIdx in 0 until seqLen - 1)
                        for (j in pastSeqLen + seqIdx + 1 until fullSeqLen) {
                            mask[seqIdx * fullSeqLen + j + maskOffset * i] += -10000f
                        }
                }
            }
            return mask
        }

        //create present state block from past + current states
        private fun MutableTypedNDArray<Any>.updateState(past: TypedNDArray<Any>?, currentState: TypedNDArray<Any>, pastBlockSize: Int, presentBlockSize: Int, i: Int, pastOffset: Int, presentOffset: Int, currentOffset: Int): Pair<MutableTypedNDArray<Any>, Int> {
            //present state block offset
            val presentStart = i * presentBlockSize + presentOffset

            var presentPos = presentStart
            if (past != null) {
                val srcPast = i * pastBlockSize + pastOffset
                this.place(presentPos, past.array, srcPast, srcPast + pastBlockSize)
                presentPos += pastBlockSize
            }
            this.place(presentPos, currentState.array, currentOffset, currentOffset + presentBlockSize - pastBlockSize)

            return this to presentStart
        }

        private fun normalizedScores(
            unidir: Boolean, queries: TypedNDArray<Any>, keys: TypedNDArray<Any>, maskIndices: TypedNDArray<Any>?, batchSize: Int,
            seqLen: Int, pastSeqLen: Int, headSize: Int, numHeads: Int, past: TypedNDArray<Any>?, present: MutableTypedNDArray<Any>
        ): TypedNDArray<Any> {
            val allSeqLen = pastSeqLen + seqLen
            val pastBlockSize = pastSeqLen * headSize
            val inputBlockSize = seqLen * headSize
            val presentBlockSize = pastBlockSize + inputBlockSize
            val scores = allocateNDArray<Any>(queries.type, Strides(intArrayOf(batchSize, numHeads, seqLen, allSeqLen)))

            val maskData = maskIndices?.maskFromIndices(unidir, batchSize, seqLen, pastSeqLen)

            val alpha = 1.0 / sqrt(headSize.toDouble())
            for (i in 0 until batchSize * numHeads) {
                val batchIdx = i / numHeads
                if (maskData != null) {
                    val start = batchIdx * seqLen * allSeqLen
                    scores.place(seqLen * allSeqLen * i, maskData.array, start, start + seqLen * allSeqLen)
                }
                val (k, kOffset) = present.updateState(past, keys, pastBlockSize, presentBlockSize, i, 0, 0, inputBlockSize * i)

                //Q*K(transposed) / sqrt(d) where d is attention head size
                gemm(queries, k, scores, inputBlockSize * i, kOffset,
                    i * seqLen * allSeqLen, seqLen, allSeqLen, headSize,
                    headSize, headSize, allSeqLen, alpha = alpha, transposeB = true)
            }
            //softmax for each result (normalize along last axis)
            return Softmax.softmax(scores, -1)
        }

        private fun attentionScore(
            scores: TypedNDArray<Any>, values: TypedNDArray<Any>, batchSize: Int, seqLen: Int, pastSeqLen: Int,
            numHeads: Int, hiddenSize: Int, past: TypedNDArray<Any>?, present: MutableTypedNDArray<Any>
        ): Pair<TypedNDArray<Any>, TypedNDArray<Any>> {
            val allSeqLen = seqLen + pastSeqLen

            val headSize = hiddenSize / numHeads
            val pastBlockSize = pastSeqLen * headSize
            val inputBlockSize = seqLen * headSize
            val presentBlockSize = pastBlockSize + inputBlockSize
            val output = allocateNDArray<Any>(scores.type, Strides(intArrayOf(batchSize, seqLen, hiddenSize)))

            val pastOffset = if (past != null) batchSize * hiddenSize * pastSeqLen else 0
            val presentOffset = batchSize * hiddenSize * allSeqLen
            val tmp = allocateNDArray<Any>(scores.type, output.strides)

            for (i in 0 until batchSize * numHeads) {
                val (v, vOffset) = present.updateState(past, values, pastBlockSize, presentBlockSize, i, pastOffset, presentOffset, i * inputBlockSize)
                val attentionOffset = seqLen * allSeqLen * i
                val tmpOffset = inputBlockSize * i
                //multiply normalized scores by value
                gemm(scores, v, tmp, attentionOffset, vOffset, tmpOffset, seqLen,
                    headSize, allSeqLen, allSeqLen, headSize, headSize, beta = 0.0)

                val batchIdx = i / numHeads
                val headIdx = i % numHeads
                var srcOffset = tmpOffset
                var dstOffset = (batchIdx * seqLen * numHeads + headIdx) * headSize
                //transpose along last two axes
                repeat(seqLen) {
                    output.place(dstOffset, tmp.array, srcOffset, srcOffset + headSize)
                    srcOffset += headSize
                    dstOffset += hiddenSize
                }
            }
            return output to present
        }

        private fun getScores(
            unidir: Boolean, q: TypedNDArray<Any>, k: TypedNDArray<Any>, v: TypedNDArray<Any>, mask: TypedNDArray<Any>?,
            past: TypedNDArray<Any>?, batchSize: Int, seqLen: Int, numHeads: Int, hiddenSize: Int
        ): Pair<TypedNDArray<Any>, TypedNDArray<Any>> {
            var pastSeqLen = 0
            val headSize = hiddenSize / numHeads
            val presentDims = intArrayOf(2, batchSize, hiddenSize / headSize, seqLen, headSize)
            if (past != null) {
                pastSeqLen = past.shape[3]
                presentDims[3] += pastSeqLen
            }

            val present = allocateNDArray<Any>(q.type, Strides(presentDims))

            val scores = normalizedScores(unidir, q, k, mask, batchSize, seqLen, pastSeqLen, headSize, hiddenSize / headSize, past, present)
            return attentionScore(scores, v, batchSize, seqLen, pastSeqLen, numHeads, hiddenSize, past, present)
        }
    }

    private val numHeads: Int by attribute("num_heads") { it: Number -> it.toInt() }
    private val unidir: Int by attribute("unidirectional") { it: Number -> it.toInt() }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val input = inputs[0]!!.data
        val weights = inputs[1]!!.data
        val bias = inputs[2]!!.data
        val maskIndices = inputs.elementAtOrNull(3)?.data
        val past = inputs.elementAtOrNull(4)?.data

        val (batchSize, seqLen, hiddenSize) = input.shape

        val (queries, keys, values) = initQueryKeyValue(input, weights, bias, batchSize, seqLen, hiddenSize, numHeads)

        val (scores, present) = getScores(unidir == 1, queries, keys, values, maskIndices, past, batchSize, seqLen, numHeads, hiddenSize)
        return listOf(scores.asTensor(), present.asTensor())
    }
}
