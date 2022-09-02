package io.kinference.tfjs.operators.layer.attention

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class QAttention(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in QAttentionVer1.VERSION.asRange() -> QAttentionVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of QAttention operator: $version")
            }
    }
}

class QAttentionVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    QAttention(name, INFO, attributes, inputs, outputs) {

    companion object {
        private val FLOATS = setOf(TensorProto.DataType.FLOAT, TensorProto.DataType.FLOAT16)
        private val BYTES = setOf(TensorProto.DataType.INT8, TensorProto.DataType.UINT8)

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

        private fun initQueryKeyValue(
            input: NumberNDArrayTFJS, weights: NumberNDArrayTFJS, bias: NumberNDArrayTFJS,
            numHeads: Int, inputZeroPoint: NumberNDArrayTFJS?,
            weightsZeroPoint: NumberNDArrayTFJS?, deqScale: NumberNDArray
        ): Array<NumberNDArrayTFJS> {
            val (batchSize, seqLen, inputHidden) = input.shape
            val headSize = inputHidden / numHeads
            val weightsWithZP = if (weightsZeroPoint != null) weights.minus(weightsZeroPoint) else weights
            val inputWithZP = if (inputZeroPoint != null) input.minus(inputZeroPoint) else input
            val weightsPrepared = (weightsWithZP as NumberNDArrayTFJS)
                .reshape(intArrayOf(inputHidden, 1, 3, numHeads, headSize))
                .transpose(intArrayOf(2, 1, 3, 0, 4))
                .broadcastTo(arrayOf(3, batchSize, numHeads, inputHidden, headSize))
            val biasPrepared = bias.reshape(intArrayOf(3, 1, numHeads, 1, headSize))
            val inputPrepared = (inputWithZP as NumberNDArrayTFJS)
                .reshape(intArrayOf(1, batchSize, 1, seqLen, inputHidden))
                .broadcastTo(arrayOf(3, batchSize, numHeads, seqLen, inputHidden))
            val output = inputPrepared.matMul(weightsPrepared).times(deqScale).plus(biasPrepared) as NumberNDArrayTFJS
            return output.unstack(0).also {
                if (weightsZeroPoint != null) weightsWithZP.close()
                if (inputZeroPoint != null) inputWithZP.close()
                closeAll(weightsPrepared, biasPrepared, inputPrepared, output)
            }
        }
    }

    private val numHeads: Int by attribute("num_heads") { it: Number -> it.toInt() }
    private val unidir: Boolean by attribute("unidirectional") { it: Number -> it.toInt() == 1 }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data as NumberNDArrayTFJS
        val weights = inputs[1]!!.data as NumberNDArrayTFJS
        val bias = inputs[2]!!.data as NumberNDArrayTFJS
        val inputScale = inputs[3]!!.data as NumberNDArrayTFJS
        val weightsScale = inputs[4]!!.data as NumberNDArrayTFJS
        val maskIndices = inputs.getOrNull(5)?.data as? NumberNDArrayTFJS
        val inputZP = inputs.getOrNull(6)?.data as? NumberNDArrayTFJS
        val weightsZP = inputs.getOrNull(7)?.data as? NumberNDArrayTFJS
        val past = inputs.getOrNull(8)?.data as? NumberNDArrayTFJS

        val (batchSize, seqLen, hiddenSize) = input.shape
        val fullScale = inputScale * weightsScale

        val (queries, keys, values) = initQueryKeyValue(input, weights, bias, numHeads, inputZP, weightsZP, fullScale)

        val outputs = Attention.getScores(unidir, queries, keys, values, maskIndices, past, batchSize, seqLen, numHeads, hiddenSize)

        return listOf(outputs[0].asTensor(), outputs[1].asTensor()).also {
            closeAll(queries, keys, values)
        }
    }
}

