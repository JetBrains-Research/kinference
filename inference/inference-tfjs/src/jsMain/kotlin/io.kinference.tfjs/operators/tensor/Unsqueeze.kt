package io.kinference.tfjs.operators.tensor

import io.kinference.ndarray.toIntArray
import io.kinference.protobuf.message.AttributeProto
import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.extensions.*
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*

class Unsqueeze(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axes", setOf(AttributeProto.AttributeType.INTS), true)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "expanded", optional = false))

        private val INFO = OperatorInfo("Unsqueeze", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val axes: IntArray by attribute { it: LongArray -> it.toIntArray() }

    override fun apply(context: Context, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidy {
            val input = inputs[0]!!.data
            val actualAxes = axes.map { input.indexAxis(it) }.sorted()
            val newShape = input.shape.toMutableList()
            for (axis in actualAxes) {
                newShape.add(axis, 1)
            }
            return@tidy arrayOf(input.reshape(newShape.toTypedArray()))
        }
        return listOf(outputs[0].asTensor("expanded"))
    }
}
