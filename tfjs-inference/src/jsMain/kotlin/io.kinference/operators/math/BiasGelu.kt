package io.kinference.operators.math

import io.kinference.attributes.Attribute
import io.kinference.custom_externals.core.scalar
import io.kinference.custom_externals.extensions.*
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.operators.*
import kotlin.math.sqrt

class BiasGelu(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {

    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val SQRT2 = scalar(sqrt(2.0f), "float32")
        private val scalarOne = scalar(1.0f, "float32")
        private val scalarHalfOne = scalar(0.5f, "float32")


        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false)
        )

        private val INFO = OperatorInfo("BiasGelu", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)
    }


    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val outputs = tidy {
            val input = inputs[0]!!.data
            val bias = inputs[1]!!.data
            val sum = input + bias
            val erfSum = (sum / SQRT2).erf()
            val output = (erfSum + scalarOne) * scalarHalfOne * sum
            return@tidy arrayOf(output)
        }
        return listOf(outputs[0].asTensor("C"))
    }
}
