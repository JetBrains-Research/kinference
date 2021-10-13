package io.kinference.tfjs.operators.math

import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.core.scalar
import io.kinference.tfjs.externals.extensions.*
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*

class FastGelu(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val COEF_1 = scalar(0.5f, "float32")
        private val COEF_2 = scalar(0.035677408136300125f, "float32")
        private val COEF_3 = scalar(0.7978845608028654f, "float32")

        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "bias", optional = true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false)
        )

        private val INFO = OperatorInfo("FastGelu", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)
    }


    override fun apply(context: Context, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidy {
            val input = inputs.first()!!.data
            val bias = inputs.getOrNull(1)?.data

            val inputWithBias = if (bias != null) input + bias else input

            val output = inputWithBias * (COEF_1 + COEF_1 * tanh(inputWithBias * (COEF_2 * inputWithBias * inputWithBias + COEF_3)))
            return@tidy arrayOf(output)
        }

        return listOf(outputs[0].asTensor("Y"))
    }
}

