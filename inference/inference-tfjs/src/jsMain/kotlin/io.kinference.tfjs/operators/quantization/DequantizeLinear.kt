package io.kinference.tfjs.operators.quantization

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class DequantizeLinear(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 10)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in DequantizeLinearVer10.VERSION.asRange() -> DequantizeLinearVer10(name, attributes, inputs, outputs)
                else -> error("Unsupported version of DequantizeLinear operator: $version")
            }
    }
}

class DequantizeLinearVer10(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    DequantizeLinear(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val IN_TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.INT8,
            TensorProto.DataType.UINT8
        )

        private val OUT_TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), required = false, default = 1)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, IN_TYPE_CONSTRAINTS, "x", optional = false),
            IOInfo(0, OUT_TYPE_CONSTRAINTS, "x_scale", optional = false),
            IOInfo(0, IN_TYPE_CONSTRAINTS, "x_zero_point", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, OUT_TYPE_CONSTRAINTS, "y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 10)
        private val INFO = OperatorInfo("DequantizeLinear", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }


    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data as NumberNDArrayTFJS
        val scale = inputs[1]!!.data as NumberNDArrayTFJS
        val zeroPoint = inputs.getOrNull(2)?.data as? NumberNDArrayTFJS
        val actualAxis = input.indexAxis(axis)

        require(zeroPoint == null || scale.shape.contentEquals(zeroPoint.shape)) { "Zero point and scale tensors should have the same dims" }

        val output = tidyNDArray {
            return@tidyNDArray when {
                canDequantizePerTensor(zeroPoint, scale) -> {
                    val zero = zeroPoint ?: NDArrayTFJS.intScalar(0)
                    (input - zero) * scale
                }

                input.canDequantizePerAxis(actualAxis, zeroPoint, scale) -> {
                    val blockCount = input.computeBlockSize(toDim = actualAxis)
                    val blockSize = input.computeBlockSize(fromDim = actualAxis + 1)
                    val dim = input.shape[actualAxis]
                    val preparedInput = input.reshape(intArrayOf(blockCount, dim, blockSize))
                    val preparedZP = zeroPoint?.reshape(intArrayOf(1, dim, 1)) ?: NDArrayTFJS.intZeros(arrayOf(1, dim, 1))
                    val preparedScale = scale.reshape(intArrayOf(1, dim, 1))

                    val rawOutput = (preparedInput - preparedZP) * preparedScale
                    rawOutput.reshape(input.shape)
                }

                else -> error("Cannot perform dequantization. Scale and zero point tensors should be either scalars or 1D tensors")
            }
        }
        return listOf(output.asTensor("y"))
    }
}

