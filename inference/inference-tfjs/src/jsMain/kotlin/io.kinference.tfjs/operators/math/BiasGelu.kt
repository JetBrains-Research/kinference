package io.kinference.tfjs.operators.math

import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.core.scalar
import io.kinference.tfjs.externals.extensions.*
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*
import kotlin.math.sqrt

sealed class BiasGelu(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<TFJSTensor, TFJSTensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in BiasGeluVer1.VERSION.asRange() -> BiasGeluVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of BiasGelu operator: $version")
        }
    }
}

class BiasGeluVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(INFO, attributes, inputs, outputs) {

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

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("BiasGelu", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")
    }


    override fun apply(context: Context, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
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
