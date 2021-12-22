package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.concatenate
import io.kinference.core.graph.KIContext
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.operator.*
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto

sealed class Concat(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 4)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ConcatVer4.VERSION.asRange() -> ConcatVer4(attributes, inputs, outputs)
            else -> error("Unsupported version of Concat operator: $version")
        }
    }
}

@ExperimentalTime
class ConcatVer4(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Concat(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), true)
        )

        private val INPUTS_INFO = listOf(VariadicIOInfo(0, TYPE_CONSTRAINTS, "inputs", minimumArity = 1, differentiable = true, heterogeneous = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "concat_result", optional = false, differentiable = true))

        internal val VERSION = VersionInfo(sinceVersion = 4)
        private val INFO = OperatorInfo("Concat", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<KITensor?>, profilingContext: ProfilingContext?, checkCancelled: () -> Unit): List<KITensor?> {
        return listOf(inputs.requireNoNulls().concatenate(axis))
    }
}
