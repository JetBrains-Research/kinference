package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.memory.ManualAllocatorContext
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import kotlin.coroutines.coroutineContext

sealed class MatMul(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in MatMulVer1.VERSION.asRange() -> MatMulVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of MatMul operator: $version")
        }
    }
}


class MatMulVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : MatMul(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.BFLOAT16
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("MatMul", emptySet(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val manualContext = coroutineContext[ManualAllocatorContext.Key]

        val first = inputs[0]!!.data as NumberNDArrayCore
        val second = inputs[1]!!.data as NumberNDArrayCore

        val destShape = Broadcasting.broadcastShapeForMatmul(first.shape, second.shape)
        val destStrides = Strides(destShape)

        val dest = (manualContext?.getNDArray(first.type, destStrides, fillZeros = true) ?: allocateNDArray(first.type, destStrides)) as MutableNumberNDArrayCore

        return listOf((first.matmul(second, dest)).asTensor("Y", manualContext))
    }
}
