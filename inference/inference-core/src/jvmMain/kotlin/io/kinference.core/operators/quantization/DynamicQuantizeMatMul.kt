package io.kinference.core.operators.quantization

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.kinference.ndarray.extensions.tryDequantize
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto

sealed class DynamicQuantizeMatMul(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in DynamicQuantizeMatMulVer11.VERSION.asRange() -> DynamicQuantizeMatMulVer11(name, attributes, inputs, outputs)
            else -> error("Unsupported version of DynamicQuantizeMatMul operator: $version")
        }
    }
}


class DynamicQuantizeMatMulVer11(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : DynamicQuantizeMatMul(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val BYTES_TYPES = setOf(
            TensorProto.DataType.INT8,
            TensorProto.DataType.UINT8
        )

        private val FLOAT_TYPE = setOf(TensorProto.DataType.FLOAT)

        private val INPUTS_INFO = listOf(
            IOInfo(0, FLOAT_TYPE, "A", optional = false),
            IOInfo(1, BYTES_TYPES, "B", optional = false),
            IOInfo(2, FLOAT_TYPE, "b_scale", optional = false),
            IOInfo(3, BYTES_TYPES, "b_zero_point", optional = true),
            IOInfo(4, FLOAT_TYPE, "bias", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, FLOAT_TYPE, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("DynamicQuantizeMatMul", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.ORT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val left = inputs[0]!!.data as FloatNDArray
        val quantizedRight = inputs[1]!!.data as NumberNDArrayCore

        val rightScale = inputs[2]!!.data as FloatNDArray
        val rightZeroPoint = inputs[3]?.data as? NumberNDArrayCore
        val bias = inputs[4]?.data as? FloatNDArray

        val dequantRight = quantizedRight.tryDequantize(rightZeroPoint, rightScale)

        val output = left.matmul(dequantRight)

        if (bias != null) {
            output.plusAssign(bias)
        }

        return listOf(output.asTensor("Y"))
    }
}
