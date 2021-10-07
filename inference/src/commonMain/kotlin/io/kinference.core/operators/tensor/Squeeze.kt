package io.kinference.core.operators.tensor

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensors.KITensor
import io.kinference.core.data.tensors.asTensor
import io.kinference.core.graph.Context
import io.kinference.core.graph.ProfilingContext
import io.kinference.ndarray.extensions.squeeze
import io.kinference.ndarray.toIntArray
import io.kinference.core.operators.*
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto

@ExperimentalTime
class Squeeze(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axes", setOf(AttributeProto.AttributeType.INTS), false)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "squeezed", optional = false))

        private val INFO = OperatorInfo("Squeeze", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val axes: LongArray? by attributeOrNull()

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val squeezeAxes = axes?.toIntArray() ?: IntArray(0)
        return listOf(inputs.first()!!.data.toMutable().squeeze(*squeezeAxes).asTensor())
    }
}
