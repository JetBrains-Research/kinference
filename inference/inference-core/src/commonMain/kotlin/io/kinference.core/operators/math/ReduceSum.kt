package io.kinference.core.operators.math

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.core.operators.*
import io.kinference.ndarray.arrays.NumberNDArray
import io.kinference.ndarray.toIntArray
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

sealed class ReduceSum(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ReduceSumVer1.VERSION.asRange() -> ReduceSumVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of ReduceSum operator: $version")
        }
    }
}

@ExperimentalTime
class ReduceSumVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : ReduceSum(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false),
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "reduced", optional = false)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axes", setOf(AttributeProto.AttributeType.INTS), false, LongArray(0)),
            AttributeInfo("keepdims", setOf(AttributeProto.AttributeType.INT), false, 1),
        )

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)
        private val INFO = OperatorInfo("ReduceSum", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axes: LongArray by attribute()
    private val keepDims: Boolean by attribute("keepdims") { it: Long -> it == 1L }

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs.first()!!.data as NumberNDArray
        val actualAxes = if (axes.isEmpty()) input.shape.indices.toIntArray() else axes.toIntArray()
        return listOf(input.reduceSum(actualAxes, keepDims).asTensor("reduced"))
    }
}
