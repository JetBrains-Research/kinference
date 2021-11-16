package io.kinference.core.operators.tensor

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.extensions.reshape
import io.kinference.core.operators.*
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

sealed class Reshape(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in ReshapeVer5.VERSION.asRange() -> ReshapeVer5(attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

@ExperimentalTime
class ReshapeVer5(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Reshape(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "reshaped", optional = false, differentiable = true))

        internal val VERSION = VersionInfo(sinceVersion = 5, untilVersion = 14)
        private val INFO = OperatorInfo("Reshape", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val targetShape = inputs[1]!!.data
        return listOf(inputs[0]!!.data.toMutable().reshape(targetShape).asTensor())
    }
}
