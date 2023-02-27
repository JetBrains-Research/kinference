package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

sealed class MatMulIntegerToFloat(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in MatMulIntegerToFloatVer1.VERSION.asRange() -> MatMulIntegerToFloatVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of MatMulIntegerToFloat operator: $version")
        }
    }
}

@ExperimentalTime
class MatMulIntegerToFloatVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : MatMulIntegerToFloat(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val IN_TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT8,
            TensorProto.DataType.INT8
        )

        private val OUT_TYPE_CONSTRAINTS = setOf(TensorProto.DataType.FLOAT)

        private val INPUTS_INFO = listOf(
            IOInfo(0, IN_TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, IN_TYPE_CONSTRAINTS, "B", optional = false),
            IOInfo(2, OUT_TYPE_CONSTRAINTS, "a_scale", optional = false),
            IOInfo(3, OUT_TYPE_CONSTRAINTS, "b_scale", optional = false),
            IOInfo(4, IN_TYPE_CONSTRAINTS, "a_zero_point", optional = true),
            IOInfo(5, IN_TYPE_CONSTRAINTS, "b_zero_point", optional = true),
            IOInfo(6, OUT_TYPE_CONSTRAINTS, "bias", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, OUT_TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("MatMulIntegerToFloat", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val left = inputs[0]!!.data as NumberNDArrayCore
        val right = inputs[1]!!.data as NumberNDArrayCore
        val leftScale = inputs[2]!!.data as FloatNDArray
        val rightScale = inputs[3]!!.data as FloatNDArray

        val leftZeroPoint = inputs[4]?.data as? NumberNDArrayCore
        val rightZeroPoint = inputs[5]?.data as? NumberNDArrayCore

        val bias = inputs[6]?.data as? FloatNDArray

        val leftDequant = left.tryDequantize(leftZeroPoint, leftScale)
        val rightDequant = right.tryDequantize(rightZeroPoint, rightScale)

        val outputArray = leftDequant.matmul(rightDequant)

        if (bias != null) {
            outputArray.plusAssign(bias)
        }

        return listOf(outputArray.asTensor("Y"))
    }
}

