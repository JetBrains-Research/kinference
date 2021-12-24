package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.acceptWithRecursive
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import kotlin.math.*
import kotlin.time.ExperimentalTime

sealed class BiasGelu(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in BiasGeluVer1.VERSION.asRange() -> BiasGeluVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of BiasGelu operator: $version")
        }
    }
}

@ExperimentalTime
class BiasGeluVer1(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : BiasGelu(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("BiasGelu", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")

        private val SQRT_2_1_FLOAT = 1f / sqrt(2f)
        private val SQRT_2_1 = 1.0 / sqrt(2.0)
    }


    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data
        val bias = inputs[1]!!.data

        return when(input.type) {
            DataType.FLOAT -> {
                input as FloatNDArray
                bias as FloatNDArray

                val output = input.allocateNDArray(input.strides)

                val inputPointer = input.array.pointer()
                val outputPointer = output.array.pointer()
                val biasPointer = bias.array.pointer()
                outputPointer.acceptWithRecursive(inputPointer, biasPointer, input.linearSize) { _: Float, input: Float, bias: Float ->
                    val value = input + bias

                    val valueForErf = value * SQRT_2_1_FLOAT
                    val sign = valueForErf.sign
                    val absValue = abs(valueForErf)
                    val t = 1f / (1f + ERF_P_VALUE_FLOAT * absValue)
                    val sum = t * (ERF_COEF_1_FLOAT + t * (ERF_COEF_2_FLOAT + t * (ERF_COEF_3_FLOAT + t * (ERF_COEF_4_FLOAT + t * ERF_COEF_5_FLOAT))))
                    val erfValue = sign * (1f - sum * exp(-absValue * absValue))
                    0.5f * value * (1.0f + erfValue)
                }

                listOf(output.asTensor("C"))
            }

            DataType.DOUBLE -> {
                input as DoubleNDArray
                bias as DoubleNDArray

                val output = input.allocateNDArray(input.strides)

                val inputPointer = input.array.pointer()
                val outputPointer = output.array.pointer()
                val biasPointer = bias.array.pointer()
                outputPointer.acceptWithRecursive(inputPointer, biasPointer, input.linearSize) { _: Double, input: Double, bias: Double ->
                    val value = input + bias

                    val valueForErf = value * SQRT_2_1
                    val sign = valueForErf.sign
                    val absValue = abs(valueForErf)
                    val t = 1.0 / (1.0 + ERF_P_VALUE * absValue)
                    val sum = t * (ERF_COEF_1 + t * (ERF_COEF_2 + t * (ERF_COEF_3 + t * (ERF_COEF_4 + t * ERF_COEF_5))))
                    val erfValue = sign * (1.0 - sum * exp(-absValue * absValue))

                    0.5 * value * (1.0 + erfValue)
                }

                listOf(output.asTensor("C"))
            }

            else -> error("")
        }
    }
}
