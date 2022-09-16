package io.kinference.tfjs.operators.layer.attention

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.*
import kotlin.math.min
import kotlin.math.sqrt

sealed class Attention(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {

    companion object {
        internal fun NDArrayTFJS?.maskFromIndices(unidir: Boolean, batchSize: Int, seqLen: Int, pastSeqLen: Int): NumberNDArrayTFJS {
            return tidyNDArray {
                val fullSeqLen = pastSeqLen + seqLen
                val output = when {
                    this != null && this.rank == 1 -> {
                        val maskIndices = this.dataInt()
                        val outputArray = FloatArray(batchSize * fullSeqLen)
                        repeat(batchSize) { batch ->
                            val startIdx = maskIndices[batch]
                            val batchStartIdx = fullSeqLen * batch
                            outputArray.fill(-10000f, batchStartIdx + startIdx, batchStartIdx + fullSeqLen)

                            if (this.shape[0] == 2 * batchSize) {
                                val endIdx = maskIndices[batch + batchSize]
                                outputArray.fill(-10000f, batchStartIdx, batchStartIdx + min(endIdx, fullSeqLen))
                            }
                        }
                        NDArrayTFJS.float(outputArray, arrayOf(batchSize, 1, fullSeqLen))
                    }

                    this != null && this.rank == 2 -> {
                        val maskIndices = this.dataInt()
                        val outputArray = FloatArray(batchSize * fullSeqLen)
                        for (idx in maskIndices.indices) {
                            val src = maskIndices[idx]
                            outputArray[idx] = if (src > 0) 0f else -10000f
                        }
                        NDArrayTFJS.float(outputArray, arrayOf(batchSize, 1, fullSeqLen))
                    }

                    else -> error("Unsupported mask")
                }

                val broadcastedOutput = output.broadcastTo(arrayOf(batchSize, seqLen, fullSeqLen))

                val outputWithUnidir = if (unidir) {
                    val outputData = broadcastedOutput.dataFloat()
                    repeat(batchSize) { batch ->
                        repeat(seqLen - 1) { seqIdx ->
                            val startIdx = pastSeqLen + seqIdx + 1
                            val offsetIdx = seqIdx * fullSeqLen + batch * seqLen * fullSeqLen
                            for (idx in offsetIdx + startIdx until offsetIdx + fullSeqLen) {
                                outputData[idx] -= 10000f
                            }
                        }
                    }
                    NDArrayTFJS.float(outputData, broadcastedOutput.shapeArray)
                } else broadcastedOutput
                return@tidyNDArray outputWithUnidir
            } as NumberNDArrayTFJS
        }

        internal fun normalizedScores(
            unidir: Boolean, queries: NumberNDArrayTFJS, maskIndices: NumberNDArrayTFJS?, batchSize: Int,
            seqLen: Int, pastSeqLen: Int, headSize: Int, present: NumberNDArrayTFJS
        ): NumberNDArrayTFJS {
            return tidyNDArray {
                val fullSeqLen = pastSeqLen + seqLen
                val maskData = maskIndices.maskFromIndices(unidir, batchSize, seqLen, pastSeqLen).reshape(intArrayOf(batchSize, 1, seqLen, fullSeqLen))

                val alpha = NDArrayTFJS.floatScalar(1.0f / sqrt(headSize.toFloat()))
                val scoreData = queries.matmul(present, transposeRight = true).times(alpha).plus(maskData)

                return@tidyNDArray scoreData.softmax(-1)
            } as NumberNDArrayTFJS
        }

        internal fun attentionScore(
            scores: NumberNDArrayTFJS, batchSize: Int, seqLen: Int,
            hiddenSize: Int, present: NumberNDArrayTFJS
        ): NDArrayTFJS {
            return tidyNDArray {
                val output = scores.matmul(present)
                val newShape = intArrayOf(batchSize, seqLen, hiddenSize)
                return@tidyNDArray output.transpose(intArrayOf(0, 2, 1, 3)).reshape(newShape)
            }
        }

        internal fun getScores(
            unidir: Boolean, q: NumberNDArrayTFJS, k: NumberNDArrayTFJS, v: NumberNDArrayTFJS, mask: NumberNDArrayTFJS?,
            past: NumberNDArrayTFJS?, batchSize: Int, seqLen: Int, numHeads: Int, hiddenSize: Int
        ): Array<NDArrayTFJS> {
            return tidyNDArrays {
                val present = k.stack(listOf(v), axis = 0)
                val pastSeqLen = if (past != null) past.shape[3] else 0
                val headSize = hiddenSize / numHeads
                val presentWithPast = past?.concat(listOf(present), axis = 3) ?: present
                val (presentKeys, presentValue) = if (past == null) arrayOf(k, v) else (presentWithPast as NumberNDArrayTFJS).unstack(0)
                val scores = normalizedScores(unidir, q, mask, batchSize, seqLen, pastSeqLen, headSize, presentKeys)
                return@tidyNDArrays arrayOf(attentionScore(scores, batchSize, seqLen, hiddenSize, presentValue), presentWithPast)
            }
        }

        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in AttentionVer1.VERSION.asRange() -> AttentionVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Attention operator: $version")
            }
    }
}

class AttentionVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Attention(name, INFO, attributes, inputs, outputs) {
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

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("Attention", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")

        @Suppress("UNCHECKED_CAST")
        internal fun initQueryKeyValue(input: NumberNDArrayTFJS, weights: NumberNDArrayTFJS, bias: NumberNDArrayTFJS, numHeads: Int): Array<NumberNDArrayTFJS> {
            return tidyNDArrays {
                val (batchSize, seqLen, inputHidden) = input.shape
                val headSize = inputHidden / numHeads
                val weightsPrepared = weights
                    .reshape(intArrayOf(inputHidden, 1, 3, numHeads, headSize))
                    .transpose(intArrayOf(2, 1, 3, 0, 4))
                    .broadcastTo(arrayOf(3, batchSize, numHeads, inputHidden, headSize))
                val biasPrepared = bias.reshape(intArrayOf(3, 1, numHeads, 1, headSize))
                val inputPrepared = input
                    .reshape(intArrayOf(1, batchSize, 1, seqLen, inputHidden))
                    .broadcastTo(arrayOf(3, batchSize, numHeads, seqLen, inputHidden))

                val output = inputPrepared.matmul(weightsPrepared).plus(biasPrepared)
                return@tidyNDArrays output.unstack(0)
            } as Array<NumberNDArrayTFJS>
        }
    }

    private val numHeads: Int by attribute("num_heads") { it: Number -> it.toInt() }
    private val unidir: Boolean by attribute("unidirectional") { it: Number -> it.toInt() == 1 }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data as NumberNDArrayTFJS
        val weights = inputs[1]!!.data as NumberNDArrayTFJS
        val bias = inputs[2]!!.data as NumberNDArrayTFJS
        val maskIndices = inputs.elementAtOrNull(3)?.data as? NumberNDArrayTFJS
        val past = inputs.elementAtOrNull(4)?.data as? NumberNDArrayTFJS

        val (batchSize, seqLen, hiddenSize) = input.shape

        val outputs = tidyNDArrays {
            val (queries, keys, values) = initQueryKeyValue(input, weights, bias, numHeads)
            return@tidyNDArrays getScores(unidir, queries, keys, values, maskIndices, past, batchSize, seqLen, numHeads, hiddenSize)
        }

        return outputs.asNamedOutputs(this.outputs)
    }
}
