package io.kinference.tfjs.operators.quantization

import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.custom_externals.core.scalar
import io.kinference.tfjs.custom_externals.extensions.*
import io.kinference.tfjs.data.tensors.Tensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*
import io.kinference.protobuf.message.TensorProto

class DynamicQuantizeLinear(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val byteSizeScalar = scalar(255f, "float32")

        private val scalarZero = scalar(0f, "float32")

        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.FLOAT), "x", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.UINT8), "y", optional = false),
            IOInfo(1, setOf(TensorProto.DataType.FLOAT), "y_scale", optional = false),
            IOInfo(2, setOf(TensorProto.DataType.UINT8), "y_zero_point", optional = false)
        )

        private val INFO = OperatorInfo("DynamicQuantizeLinear", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val outputs = tidy {
            val input = inputs[0]!!.data

            val inputMin = min(input.min(), scalarZero)
            val inputMax = max(input.max(), scalarZero)

            val outputScale = (inputMax - inputMin) / byteSizeScalar

            val outputZeroPoint = (-inputMin / outputScale).round().clip(0f, 255f).cast("int32")

            val quantInput = ((input / outputScale).round() + outputZeroPoint).clip(0f, 255f).cast("int32")

            return@tidy arrayOf(quantInput, outputScale, outputZeroPoint)
        }

        return listOf(outputs[0].asTensor("y"), outputs[1].asTensor("y_scale"), outputs[2].asTensor("y_zero_point"))
    }
}

