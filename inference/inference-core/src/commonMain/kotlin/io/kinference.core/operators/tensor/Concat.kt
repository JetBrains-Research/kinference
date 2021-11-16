package io.kinference.core.operators.tensor

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.concatenate
import io.kinference.core.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.core.operators.*
import io.kinference.core.operators.VersionInfo.Companion.asRange
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto

sealed class Concat(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
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

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        return listOf(inputs.requireNoNulls().concatenate(axis))
    }
}
