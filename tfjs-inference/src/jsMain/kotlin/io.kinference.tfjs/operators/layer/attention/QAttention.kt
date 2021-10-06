package io.kinference.tfjs.operators.layer.attention

import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.custom_externals.core.TensorTFJS
import io.kinference.tfjs.custom_externals.extensions.*
import io.kinference.tfjs.data.tensors.Tensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

class QAttention(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {

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

        private val INFO = OperatorInfo("QAttention", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)

        private fun initQueryKeyValue(input: TensorTFJS, weights: TensorTFJS, bias: TensorTFJS,
                                      numHeads: Int, inputZeroPoint: TensorTFJS?,
                                      weightsZeroPoint: TensorTFJS?, deqScale: TensorTFJS): Array<TensorTFJS> {
            return tidy {
                val (batchSize, seqLen, inputHidden) = input.shape
                val headSize = inputHidden / numHeads
                val weightsWithZP = if (weightsZeroPoint != null) weights.minus(weightsZeroPoint) else weights
                val inputWithZP = if (inputZeroPoint != null) input.minus(inputZeroPoint) else input
                val weightsPrepared = weightsWithZP
                    .reshape(arrayOf(inputHidden, 1, 3, numHeads, headSize))
                    .transpose(arrayOf(2, 1, 3, 0, 4))
                    .broadcastTo(arrayOf(3, batchSize, numHeads, inputHidden, headSize))
                val biasPrepared = bias.reshape(arrayOf(3, 1, numHeads, 1, headSize))
                val inputPrepared = inputWithZP
                    .reshape(arrayOf(1, batchSize, 1, seqLen, inputHidden))
                    .broadcastTo(arrayOf(3, batchSize, numHeads, seqLen, inputHidden))
                val output = inputPrepared.matMul(weightsPrepared).times(deqScale).plus(biasPrepared)
                return@tidy output.unstack(0)
            }
        }
    }

    private val numHeads: Int by attribute("num_heads") { it: Number -> it.toInt() }
    private val unidir: Boolean by attribute("unidirectional") { it: Number -> it.toInt() == 1 }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val outputs = tidy {
            val input = inputs[0]!!.data
            val weights = inputs[1]!!.data
            val bias = inputs[2]!!.data
            val inputScale = inputs[3]!!.data
            val weightsScale = inputs[4]!!.data
            val maskIndices = inputs.getOrNull(5)?.data
            val inputZP = inputs.getOrNull(6)?.data
            val weightsZP = inputs.getOrNull(7)?.data
            val past = inputs.getOrNull(8)?.data

            val (batchSize, seqLen, hiddenSize) = input.shape


            val fullScale = inputScale * weightsScale

            val (queries, keys,  values) = initQueryKeyValue(input, weights, bias, numHeads, inputZP, weightsZP, fullScale)

            return@tidy Attention.getScores(unidir, queries, keys, values, maskIndices, past, batchSize, seqLen, numHeads, hiddenSize)
        }

        return listOf(outputs[0].asTensor(), outputs[1].asTensor())
    }
}

