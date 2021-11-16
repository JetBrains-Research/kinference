package io.kinference.core.operators.math

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.graph.Context
import io.kinference.core.operators.*
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

sealed class Div(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in DivVer7.VERSION.asRange() -> DivVer7(attributes, inputs, outputs)
            else -> error("Unsupported version of Div operator: $version")
        }
    }
}

@ExperimentalTime
class DivVer7(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Div(INFO, attributes, inputs, outputs) {
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
            IOInfo(0, TYPE_CONSTRAINTS, "A", differentiable = true, optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", differentiable = true, optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "C", differentiable = true, optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 7)
        private val INFO = OperatorInfo("Div", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val result = inputs[0]!! / inputs[1]!!
        return listOf(result.rename("C"))
    }
}
