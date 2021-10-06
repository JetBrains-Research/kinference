package io.kinference.tfjs.operators.layer.normalization

import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.custom_externals.core.scalar
import io.kinference.tfjs.custom_externals.extensions.*
import io.kinference.tfjs.data.tensors.Tensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

class LayerNormalization(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, -1),
            AttributeInfo("epsilon", setOf(AttributeProto.AttributeType.FLOAT), false, 0.00001f)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", false),
            IOInfo(1, TYPE_CONSTRAINTS, "scale", false),
            IOInfo(2, TYPE_CONSTRAINTS, "B", false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", false),
            //Unsupported
            IOInfo(1, TYPE_CONSTRAINTS, "mean", true),
            IOInfo(2, TYPE_CONSTRAINTS, "inv_std_var", true)
        )

        private val INFO = OperatorInfo("LayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }
    private val epsilon: Float by attribute()

    private val epsilonScalar = scalar(epsilon, "float32")


    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val outputs = tidy {
            val input = inputs[0]!!.data
            val scale = inputs[1]!!.data
            val bias = inputs[2]!!.data

            val actualAxis = input.indexAxis(axis)
            val axesForMoments = Array(input.rank - actualAxis) { actualAxis + it }

            val momentsOutput = input.moments(axesForMoments, keepDims = true)
            val mean = momentsOutput.mean
            val variance = momentsOutput.variance

            val normalizedInput = (input - mean) / (sqrt(variance + epsilonScalar)) * scale + bias

            return@tidy arrayOf(normalizedInput)
        }


        return listOf(outputs[0].asTensor("Y"))
    }
}

