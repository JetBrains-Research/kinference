package io.kinference.tfjs.operators.tensor

import io.kinference.protobuf.message.AttributeProto
import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.extensions.*
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*

class Concat(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<TFJSTensor, TFJSTensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), true)
        )

        private val INPUTS_INFO = listOf(VariadicIOInfo(0, TYPE_CONSTRAINTS, "inputs", minimumArity = 1, differentiable = true, heterogeneous = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "concat_result", optional = false, differentiable = true))

        private val INFO = OperatorInfo("Concat", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    override fun apply(context: Context, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidy {
            val inputsNotNull = inputs.requireNoNulls()
            val actualAxis = inputsNotNull.first().data.indexAxis(axis)
            val tensorsList = inputsNotNull.map { it.data }.toTypedArray()
            return@tidy arrayOf(tensorsList.concat(actualAxis))
        }

        return listOf(outputs[0].asTensor("concat_result"))
    }
}
