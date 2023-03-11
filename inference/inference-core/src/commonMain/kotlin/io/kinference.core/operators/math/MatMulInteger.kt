package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.ContextPrepare
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.graph.GraphContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.mapTo
import io.kinference.ndarray.arrays.tiled.IntTiledArray
import io.kinference.ndarray.extensions.tryZeroPoint
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto

sealed class MatMulInteger(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 10)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in MatMulIntegerVer10.VERSION.asRange() -> MatMulIntegerVer10(name, attributes, inputs, outputs)
            else -> error("Unsupported version of MatMulInteger operator: $version")
        }
    }
}


class MatMulIntegerVer10(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : MatMulInteger(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val IN_TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT8,
            TensorProto.DataType.INT8
        )

        private val OUT_TYPE_CONSTRAINTS = setOf(TensorProto.DataType.INT32)

        private val INPUTS_INFO = listOf(
            IOInfo(0, IN_TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, IN_TYPE_CONSTRAINTS, "B", optional = false),
            IOInfo(2, IN_TYPE_CONSTRAINTS, "a_zero_point", optional = true),
            IOInfo(3, IN_TYPE_CONSTRAINTS, "b_zero_point", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, OUT_TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 10)
        private val INFO = OperatorInfo("MatMulInteger", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        private fun NumberNDArray.toIntNDArray(): IntNDArray {
            val result = IntNDArray(IntTiledArray(this.strides), strides)
            when (this) {
                is UByteNDArray -> {
                    this.array.pointer().mapTo(result.array.pointer(), linearSize) { it.toInt() }
                }
                is ByteNDArray -> {
                    this.array.pointer().mapTo(result.array.pointer(), linearSize) { it.toInt() }
                }
                else -> error("Unsupported data type: $type")
            }

            return result
        }
    }

    object MatMulIntegerPrepare : ContextPrepare() {
        override suspend fun appendContext(context: GraphContext<KIONNXData<*>>, initializers: List<KITensor>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
            val leftTensor = initTensorByDefaultName("A", operator, initializers)
            val rightTensor = initTensorByDefaultName("B", operator, initializers)
            val leftZeroPoint = initTensorByDefaultName("a_zero_point", operator, initializers)
            val rightZeroPoint = initTensorByDefaultName("b_zero_point", operator, initializers)

            appendTensor(leftTensor, leftZeroPoint, context)
            appendTensor(rightTensor, rightZeroPoint, context)
        }

        internal suspend fun prepareTensor(tensor: KITensor, zeroPoint: KITensor?): KITensor {
            val preparedTensor = if (zeroPoint == null)
                (tensor.data as NumberNDArrayCore).toIntNDArray()
            else
                (tensor.data as NumberNDArrayCore).tryZeroPoint(zeroPoint.data as NumberNDArrayCore)

            return preparedTensor.asTensor("prepared_${tensor.name}")
        }

        private suspend fun appendTensor(tensor: KITensor?, zeroPoint: KITensor?, context: GraphContext<KIONNXData<*>>) {
            if (tensor != null) {
                val preparedTensor = prepareTensor(tensor, zeroPoint)
                context.putValue(preparedTensor.name!!, preparedTensor)
            }
        }
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val first = inputs[0]!!
        val second = inputs[1]!!
        val firstZero = inputs.getOrNull(2)
        val secondZero = inputs.getOrNull(3)

        val firstPrepared = (contexts.graph!!.getOrNullValue("prepared_${first.name}") ?: MatMulIntegerPrepare.prepareTensor(first, firstZero)) as KITensor
        val secondPrepared = (contexts.graph!!.getOrNullValue("prepared_${second.name}") ?: MatMulIntegerPrepare.prepareTensor(second, secondZero)) as KITensor

        val output = (firstPrepared.data as NumberNDArrayCore)
            .matmul(secondPrepared.data as NumberNDArrayCore)
        return listOf(output.asTensor("y"))
    }
}
