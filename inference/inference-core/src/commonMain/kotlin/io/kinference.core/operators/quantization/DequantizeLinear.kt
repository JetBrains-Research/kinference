package io.kinference.core.operators.quantization

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.arrays.NumberNDArray
import io.kinference.core.operators.*
import io.kinference.core.operators.VersionInfo.Companion.asRange
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

@ExperimentalTime
class DequantizeLinear(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val IN_TYPE_CONSTRAINTS = setOf(TensorProto.DataType.INT8, TensorProto.DataType.UINT8)

        private val OUT_TYPE_CONSTRAINTS = setOf(TensorProto.DataType.FLOAT, TensorProto.DataType.FLOAT16)

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), required = false, default = 1)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, IN_TYPE_CONSTRAINTS, "x", optional = false),
            IOInfo(0, OUT_TYPE_CONSTRAINTS, "x_scale", optional = false),
            IOInfo(0, IN_TYPE_CONSTRAINTS, "x_zero_point", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, OUT_TYPE_CONSTRAINTS, "y", optional = false))

        private val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("DequantizeLinear", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in VERSION.asRange() -> DequantizeLinear(attributes, inputs, outputs)
            else -> error("Unsupported version of DequantizeLinear operator: $version")
        }
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }


    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs[0]!!.data as NumberNDArray
        val scale = inputs[1]!!.data
        val zeroPoint = inputs.getOrNull(2)?.data

        require(zeroPoint == null || scale.shape.contentEquals(zeroPoint.shape)) { "Zero point and scale tensors should have the same dims" }

        return listOf(input.dequantize(zeroPoint, scale, axis).asTensor("y"))
    }
}
