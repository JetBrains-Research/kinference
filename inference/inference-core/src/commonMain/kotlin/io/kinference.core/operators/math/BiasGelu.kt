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
import io.kinference.utils.PlatformUtils
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.math.*
import kotlin.time.ExperimentalTime

sealed class BiasGelu(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in BiasGeluVer1.VERSION.asRange() -> BiasGeluVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of BiasGelu operator: $version")
        }
    }
}

@ExperimentalTime
class BiasGeluVer1(name: String, attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : BiasGelu(name, INFO, attributes, inputs, outputs) {
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

        private fun computeFloat(output: FloatNDArray, input: FloatNDArray, bias: FloatNDArray, startOffset: Int = 0, count: Int = input.linearSize - startOffset) {
            val inputPointer = input.array.pointer(startOffset)
            val outputPointer = output.array.pointer(startOffset)
            val biasPointer = bias.array.pointer()

            outputPointer.acceptWithRecursive(inputPointer, biasPointer, count) { _: Float, input: Float, bias: Float ->
                val value = input + bias

                val valueForErf = value * SQRT_2_1_FLOAT
                val sign = valueForErf.sign
                val absValue = abs(valueForErf)
                val t = 1f / (1f + ERF_P_VALUE_FLOAT * absValue)
                val sum = t * (ERF_COEF_1_FLOAT + t * (ERF_COEF_2_FLOAT + t * (ERF_COEF_3_FLOAT + t * (ERF_COEF_4_FLOAT + t * ERF_COEF_5_FLOAT))))
                val erfValue = sign * (1f - sum * exp(-absValue * absValue))
                0.5f * value * (1.0f + erfValue)
            }
        }

        private fun computeDouble(output: DoubleNDArray, input: DoubleNDArray, bias: DoubleNDArray, startOffset: Int = 0, count: Int = input.linearSize - startOffset) {
            val inputPointer = input.array.pointer(startOffset)
            val outputPointer = output.array.pointer(startOffset)
            val biasPointer = bias.array.pointer()

            outputPointer.acceptWithRecursive(inputPointer, biasPointer, count) { _: Double, input: Double, bias: Double ->
                val value = input + bias

                val valueForErf = value * SQRT_2_1
                val sign = valueForErf.sign
                val absValue = abs(valueForErf)
                val t = 1.0 / (1.0 + ERF_P_VALUE * absValue)
                val sum = t * (ERF_COEF_1 + t * (ERF_COEF_2 + t * (ERF_COEF_3 + t * (ERF_COEF_4 + t * ERF_COEF_5))))
                val erfValue = sign * (1.0 - sum * exp(-absValue * absValue))

                0.5 * value * (1.0 + erfValue)
            }
        }

        private suspend fun <T : NumberNDArrayCore> computeBatched(
            input: T,
            output: T,
            bias: T,
            batchFunc: suspend (T, T, T, Int, Int) -> Unit
        ) {
            val rowSize = bias.linearSize
            val numThreads = PlatformUtils.threads
            val numRows = input.linearSize / bias.linearSize
            val numBatches = if (numRows < numThreads) numRows else numThreads
            val batchSize = floor(numRows.toDouble() / numBatches).toInt()
            val endBatchOffsets = IntArray(numBatches) { batchSize * (it + 1) * rowSize }.apply {
                this[lastIndex] = input.linearSize
            }

            coroutineScope {
                for ((i, endOffset) in endBatchOffsets.withIndex()) {
                    launch {
                        val startOffset = i * batchSize * rowSize
                        val count = endOffset - startOffset
                        batchFunc(output, input, bias, startOffset, count)
                    }
                }
            }
        }
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data
        val bias = inputs[1]!!.data

        require(input.shape.last() == bias.shape.last()) { "Last dimensions of input and bias tensors must be equal" }

        // Uses ERF formula with fractional error less than x.xx * 10 ^ -4.
        // Algorithm 26.2.17 in Abromowitz and Stegun, Handbook of Mathematical.
        // Another possible ERF implementation (several ms faster):
        // https://github.com/apache/commons-numbers/blob/master/commons-numbers-gamma/src/main/java/org/apache/commons/numbers/gamma/BoostErf.java
        return when(val type = input.type) {
            DataType.FLOAT -> {
                input as FloatNDArray
                bias as FloatNDArray

                val output = MutableFloatNDArray(input.strides)

                if (contexts.execution?.coroutineContext != null && input.linearSize > bias.linearSize) {
                    computeBatched(input, output, bias, ::computeFloat)
                } else {
                    computeFloat(output, input, bias)
                }

                listOf(output.asTensor("C"))
            }

            DataType.DOUBLE -> {
                input as DoubleNDArray
                bias as DoubleNDArray

                val output = MutableDoubleNDArray(input.strides)

                if (contexts.execution?.coroutineContext != null) {
                    computeBatched(input, output, bias, ::computeDouble)
                } else {
                    computeDouble(output, input, bias)
                }

                listOf(output.asTensor("C"))
            }

            else -> error("Unsupported data type: $type")
        }
    }
}
