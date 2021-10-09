package io.kinference.tfjs.operators.tensor

import io.kinference.ndarray.toIntArray
import io.kinference.protobuf.message.AttributeProto
import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.extensions.squeeze
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*

class Squeeze(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<TFJSTensor, TFJSTensor>(INFO, attributes, inputs, outputs) {
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
    private val axesPrepared = axes?.toIntArray()?.toTypedArray()

    override fun apply(context: Context, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data
        return listOf(input.squeeze(axesPrepared).asTensor("squeezed"))
    }
}

