package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.memory.contexts.ManualAllocatorContext
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import kotlin.coroutines.coroutineContext

sealed class Add(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 7)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in AddVer7.VERSION.asRange() -> AddVer7(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Add operator: $version")
        }
    }
}


class AddVer7(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Add(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT8,
            TensorProto.DataType.UINT16,
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT8,
            TensorProto.DataType.INT16,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.BFLOAT16
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 7)
        private val INFO = OperatorInfo("Add", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val manualContext = coroutineContext[ManualAllocatorContext]

        val left = inputs[0]!!.data as NumberNDArrayCore
        val right = inputs[1]!!.data as NumberNDArrayCore

        val destShape = broadcastShape(listOf(left.shape, right.shape))
        val destStrides = Strides(destShape)
        val dest = (manualContext?.getNDArray(left.type, destStrides) ?: allocateNDArray(left.type, destStrides)) as MutableNumberNDArrayCore

        val result = left.plus(right, dest) //(inputs[0]!!.data as NumberNDArrayCore) + (inputs[1]!!.data as NumberNDArrayCore)
        return listOf(result.asTensor("C", manualContext))
    }
}
