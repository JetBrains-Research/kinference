package io.kinference.tfjs.operators.math

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.operator.*
import io.kinference.profiler.ProfilingContext
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.core.scalar
import io.kinference.tfjs.externals.extensions.*

sealed class FastGelu(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<TFJSTensor, TFJSTensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in FastGeluVer1.VERSION.asRange() -> FastGeluVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of FastGelu operator: $version")
        }
    }
}

class FastGeluVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : FastGelu(INFO, attributes, inputs, outputs) {
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

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("FastGelu", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")
    }


    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<TFJSTensor?>, profilingContext: ProfilingContext?, checkCancelled: () -> Unit): List<TFJSTensor?> {
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

