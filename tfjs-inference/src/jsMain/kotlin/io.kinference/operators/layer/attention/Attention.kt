package io.kinference.operators.layer.attention

import io.kinference.attributes.Attribute
import io.kinference.custom_externals.core.TensorTFJS
import io.kinference.custom_externals.core.scalar
import io.kinference.custom_externals.extensions.*
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.operators.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.math.min
import kotlin.math.sqrt

class Attention(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {

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

        internal fun initQueryKeyValue(input: TensorTFJS, weights: TensorTFJS, bias: TensorTFJS, numHeads: Int): Array<TensorTFJS> {
            return tidy {
                val (batchSize, seqLen, inputHidden) = input.shape
                val headSize = inputHidden / numHeads
                val weightsPrepared = weights
                    .reshape(arrayOf(inputHidden, 1, 3, numHeads, headSize))
                    .transpose(arrayOf(2, 1, 3, 0, 4))
                    .broadcastTo(arrayOf(3, batchSize, numHeads, inputHidden, headSize))
                val biasPrepared = bias.reshape(arrayOf(3, 1, numHeads, 1, headSize))
                val inputPrepared = input
                    .reshape(arrayOf(1, batchSize, 1, seqLen, inputHidden))
                    .broadcastTo(arrayOf(3, batchSize, numHeads, seqLen, inputHidden))

                val output = inputPrepared.matMul(weightsPrepared).plus(biasPrepared)
                return@tidy output.unstack(0)
            }
        }

        internal fun getScores(unidir: Boolean, q: TensorTFJS, k: TensorTFJS, v: TensorTFJS, mask: TensorTFJS?,
                               past: TensorTFJS?, batchSize: Int, seqLen: Int, numHeads: Int, hiddenSize: Int
        ): Array<TensorTFJS> {
            return tidy {
                val present = k.stack(v, axis = 0)
                val pastSeqLen = if (past != null) past.shape[3] else 0
                val headSize = hiddenSize / numHeads
                val presentWithPast = past?.concat(present, axis = 3) ?: present
                val (presentKeys, presentValue) = if (past == null) arrayOf(k, v) else presentWithPast.unstack(0)
                val scores = normalizedScores(unidir, q, mask, batchSize, seqLen, pastSeqLen, headSize, presentKeys)
                return@tidy arrayOf(attentionScore(scores, batchSize, seqLen, hiddenSize, presentValue), presentWithPast)
            }

        }

        internal fun TensorTFJS?.maskFromIndices(unidir: Boolean, batchSize: Int, seqLen: Int, pastSeqLen: Int): TensorTFJS {
            return tidy {
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
                        tensor(outputArray, arrayOf(batchSize, 1, fullSeqLen), "float32")
                    }

                    this != null && this.rank == 2 -> {
                        val maskIndices = this.dataInt()
                        val outputArray = FloatArray(batchSize * fullSeqLen)
                        for (idx in maskIndices.indices) {
                            val src = maskIndices[idx]
                            outputArray[idx] = if (src > 0) 0f else -10000f
                        }
                        tensor(outputArray, arrayOf(batchSize, 1, fullSeqLen), "float32")
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
                    tensor(outputData, broadcastedOutput.shape, "float32")
                } else broadcastedOutput

                return@tidy arrayOf(outputWithUnidir)
            }.first()
        }

        internal fun normalizedScores(
            unidir: Boolean, queries: TensorTFJS, maskIndices: TensorTFJS?, batchSize: Int,
            seqLen: Int, pastSeqLen: Int, headSize: Int, present: TensorTFJS
        ): TensorTFJS {
            return tidy {
                val fullSeqLen = pastSeqLen + seqLen
                val maskData = maskIndices.maskFromIndices(unidir, batchSize, seqLen, pastSeqLen).reshape(arrayOf(batchSize, 1, seqLen, fullSeqLen))

                val alpha = scalar(1.0f / sqrt(headSize.toFloat()), "float32")
                val scoreData = queries.matMul(present, transposeRight = true).times(alpha).plus(maskData)

                return@tidy arrayOf(scoreData.softmax())
            }.first()
        }

        internal fun attentionScore(
            scores: TensorTFJS, batchSize: Int, seqLen: Int,
            hiddenSize: Int, present: TensorTFJS
        ): TensorTFJS {
            return tidy {
                val output = scores.matMul(present)
                return@tidy arrayOf(output.transpose(arrayOf(0, 2, 1, 3)).reshape(arrayOf(batchSize, seqLen, hiddenSize)))
            }.first()
        }

    }

    private val numHeads: Int by attribute("num_heads") { it: Number -> it.toInt() }
    private val unidir: Boolean by attribute("unidirectional") { it: Number -> it.toInt() == 1 }


    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val outputs = tidy {
            val input = inputs[0]!!.data
            val weights = inputs[1]!!.data
            val bias = inputs[2]!!.data
            val maskIndices = inputs.elementAtOrNull(3)?.data
            val past = inputs.elementAtOrNull(4)?.data

            val (batchSize, seqLen, hiddenSize) = input.shape

            val (queries, keys, values) = initQueryKeyValue(
                input,
                weights,
                bias,
                numHeads
            )

            return@tidy getScores(unidir, queries, keys, values, maskIndices, past, batchSize, seqLen, numHeads, hiddenSize)
        }

        return listOf(outputs[0].asTensor(), outputs[1].asTensor())
    }
}
