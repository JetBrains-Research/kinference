package io.kinference.tfjs.operators.quantization

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.scalar
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class DynamicQuantizeLinear(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in DynamicQuantizeLinearVer11.VERSION.asRange() -> DynamicQuantizeLinearVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of DynamicQuantizeLinear operator: $version")
            }
    }
}

class DynamicQuantizeLinearVer11(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    DynamicQuantizeLinear(name, INFO, attributes, inputs, outputs) {
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

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("DynamicQuantizeLinear", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidy {
            val input = inputs[0]!!.data.tfjsArray

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

