package io.kinference.core.operators.logical

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.core.operators.*
import io.kinference.core.operators.VersionInfo.Companion.asRange
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

sealed class Or(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in OrVer7.VERSION.asRange() -> OrVer7(attributes, inputs, outputs)
            else -> error("Unsupported version of Or operator: $version")
        }
    }
}

@OptIn(ExperimentalTime::class)
class OrVer7(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Or(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(TensorProto.DataType.BOOL)

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 7)
        private val INFO = OperatorInfo("Or", emptySet(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val left = inputs[0]!!.data as BooleanNDArray
        val right = inputs[1]!!.data as BooleanNDArray

        val ans = left or right
        return listOf(ans.asTensor("C"))
    }
}
