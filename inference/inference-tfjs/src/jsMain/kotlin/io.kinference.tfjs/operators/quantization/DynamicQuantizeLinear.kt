package io.kinference.tfjs.operators.quantization

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.*
import io.kinference.utils.closeAll

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

    private val byteSizeScalar = NDArrayTFJS.floatScalar(255f)
    private val scalarZero = NDArrayTFJS.floatScalar(0f)

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidyNDArrays {
            val input = inputs[0]!!.data as NumberNDArrayTFJS

            val inputMin = min(input.min(keepDims = false), scalarZero)
            val inputMax = max(input.max(keepDims = false), scalarZero)

            val outputScale = (inputMax - inputMin) / byteSizeScalar

            val outputZeroPoint = (-inputMin / outputScale).round().clip(0f, 255f).castToInt()
            val outputZeroPointNumber = outputZeroPoint.singleValue().toInt()

            val quantInput = (input / outputScale).clip(0 - outputZeroPointNumber, 255 - outputZeroPointNumber).round().castToInt() + outputZeroPoint

            return@tidyNDArrays arrayOf(quantInput, outputScale, outputZeroPoint)
        }

        return outputs.asNamedOutputs(this.outputs)
    }

    override fun close() {
        super.close()
        closeAll(byteSizeScalar, scalarZero)
    }
}

