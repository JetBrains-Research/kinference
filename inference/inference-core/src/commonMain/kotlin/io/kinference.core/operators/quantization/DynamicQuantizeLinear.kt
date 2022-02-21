package io.kinference.core.operators.quantization

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.accept
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.extensions.createScalarNDArray
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.TensorProto
import kotlin.math.*
import kotlin.time.ExperimentalTime

sealed class DynamicQuantizeLinear(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private fun clip(x: Float, min: Float, max: Float) = when {
            x < min -> min
            x > max -> max
            else -> x
        }

        private fun Float.toUByte() = this.toUInt().toUByte()

        internal fun FloatNDArray.dynamicQuantize(): Triple<UByteNDArray, FloatNDArray, UByteNDArray> {
            val inputMin = min(0f, this.min())
            val inputMax = max(0f, this.max())

            val outputScale = (inputMax - inputMin) / 255f
            val outputScaleScalar = createScalarNDArray(DataType.FLOAT, outputScale)

            val outputZeroPoint = clip(round((-inputMin) / outputScale), 0f, 255f)
            val outputZeroPointScalar = createScalarNDArray(DataType.UBYTE, outputZeroPoint.toUByte())

            val output = allocateNDArray(DataType.UBYTE, this.strides) as MutableUByteNDArray

            output.array.pointer().accept(this.array.pointer(), this.linearSize) { _: UByte, src: Float ->
                clip((round(src / outputScale) + outputZeroPoint), 0f, 255f).toUByte()
            }

            return Triple(
                output as UByteNDArray,
                outputScaleScalar as FloatNDArray,
                outputZeroPointScalar as UByteNDArray
            )
        }

        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in DynamicQuantizeLinearVer11.VERSION.asRange() -> DynamicQuantizeLinearVer11(name, attributes, inputs, outputs)
            else -> error("Unsupported version of DynamicQuantizeLinear operator: $version")
        }
    }
}

@ExperimentalTime
class DynamicQuantizeLinearVer11(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : DynamicQuantizeLinear(name, INFO, attributes, inputs, outputs) {
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


    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs.first()!!.data as FloatNDArray

        val (output, outputScaleScalar, outputZeroPointScalar) = input.dynamicQuantize()

        return listOf(
            output.asTensor("y"),
            outputScaleScalar.asTensor("y_scale"),
            outputZeroPointScalar.asTensor("y_zero_point")
        )
    }
}
