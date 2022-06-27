package io.kinference.core.operators.layer.attention

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

sealed class QAttention(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in QAttentionVer1.VERSION.asRange() -> QAttentionVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of QAttention operator: $version")
        }
    }
}

@ExperimentalTime
class QAttentionVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : QAttention(name, INFO, attributes, inputs, outputs) {

    companion object {
        private val FLOATS = setOf(TensorProto.DataType.FLOAT, TensorProto.DataType.FLOAT16)
        private val BYTES = setOf(TensorProto.DataType.INT8, TensorProto.DataType.UINT8)
        private val DEFAULT_SCALE = FloatNDArray.scalar(1f)

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("num_heads", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("unidirectional", setOf(AttributeProto.AttributeType.INT), false, default = 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, BYTES, "input", optional = false),
            IOInfo(1, BYTES, "weight", optional = false),
            IOInfo(2, FLOATS, "bias", optional = false),
            IOInfo(3, FLOATS, "input_scale", optional = false),
            IOInfo(4, FLOATS, "weight_scale", optional = false),
            IOInfo(5, setOf(TensorProto.DataType.INT32), "mask_index", optional = true),
            IOInfo(6, BYTES, "input_zero_point", optional = true),
            IOInfo(7, BYTES, "weight_zero_point", optional = true),
            IOInfo(8, FLOATS, "past", optional = true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, FLOATS, "output", optional = false),
            IOInfo(1, FLOATS, "present", optional = true)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("QAttention", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")
    }

    private val numHeads: Int by attribute("num_heads") { it: Number -> it.toInt() }
    private val unidir: Boolean by attribute("unidirectional") { it: Number -> it.toInt() == 1 }

    /*private fun initQueryKeyValue(input: NumberNDArray, weights: NumberNDArray, bias: FloatNDArray,
                                  batchSize: Int, seqLen: Int, hiddenSize: Int, numHeads: Int,
                                  inputZeroPoint: Int, weightsZeroPoint: Int, deqScale: Float): Array<MutableNDArray> {
        val headSize = hiddenSize / numHeads

        val qkv = Array(3) { allocateNDArray(DataType.FLOAT, Strides(intArrayOf(batchSize, numHeads, seqLen, headSize))) }

        runBlocking(// use execution context here instead of Dispatchers.Default) {
            for (qkvIdx in 0 until 3) {
                launch {
                    val output = qkv[qkvIdx]

                    for (batchNum in 0 until batchSize) {
                        val inputMatrix = input.view(batchNum)
                        for (numHead in 0 until numHeads) {
                            val outputMatrix = output.viewMutable(batchNum, numHead) as MutableFloatNDArray
                            val weightsMatrix = weights.view(qkvIdx, numHead)
                            val biasMatrix = bias.view(qkvIdx, numHead)

                            when {
                                inputMatrix is ByteNDArray && weightsMatrix is ByteNDArray -> inputMatrix.quantizeDot(weightsMatrix, outputMatrix, inputZeroPoint, weightsZeroPoint, deqScale)
                                inputMatrix is ByteNDArray && weightsMatrix is UByteNDArray -> inputMatrix.quantizeDot(weightsMatrix, outputMatrix, inputZeroPoint, weightsZeroPoint, deqScale)
                                inputMatrix is UByteNDArray && weightsMatrix is ByteNDArray -> inputMatrix.quantizeDot(weightsMatrix, outputMatrix, inputZeroPoint, weightsZeroPoint, deqScale)
                                inputMatrix is UByteNDArray && weightsMatrix is UByteNDArray -> inputMatrix.quantizeDot(weightsMatrix, outputMatrix, inputZeroPoint, weightsZeroPoint, deqScale)
                            }
                            outputMatrix.plusAssign(biasMatrix)
                        }
                    }
                }
            }
        }

        return qkv
    }*/

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data as NumberNDArray
        val inputScale = inputs[3]!!.data as NumberNDArray
        val inputZeroPoint = inputs.getOrNull(6)?.data as NumberNDArray?
        val dequantInput = input.dequantize(inputZeroPoint, inputScale)

        val weights = inputs[1]!!
        val weightsScale = inputs[4]!!
        val weightsZeroPoint = inputs.getOrNull(7)

        val preparedWeights = (contexts.graph!!.getOrNullValue("prepared_${weights.name}") ?: QAttentionContext.prepareWeights(weights, weightsScale, weightsZeroPoint, numHeads)) as KITensor

        val bias = inputs[2]!!

        val preparedBias = (contexts.graph!!.getOrNullValue("prepared_${bias.name}") ?: AttentionContext.prepareBias(bias, numHeads)) as KITensor

        val maskIndices = inputs.getOrNull(5)?.data as IntNDArray?
        val past = inputs.getOrNull(8)?.data as NumberNDArray?

        val (batchSize, seqLen, hiddenSize) = dequantInput.shape
        val (queries, keys, values) = AttentionVer1.initQueryKeyValue(
            dequantInput,
            preparedWeights.data,
            preparedBias.data,
            batchSize, seqLen, hiddenSize, numHeads,
            contexts.execution
        )
        val (scores, present) = Attention.getScores(unidir, queries, keys, values, maskIndices, past, batchSize, seqLen, numHeads, hiddenSize, contexts.execution)
        return listOf(scores.asTensor(), present.asTensor())
    }
}
