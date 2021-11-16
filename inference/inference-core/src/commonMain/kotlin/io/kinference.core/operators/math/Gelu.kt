package io.kinference.core.operators.math

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.map
import io.kinference.core.operators.*
import io.kinference.primitives.types.DataType
import kotlin.math.sqrt
import kotlin.time.ExperimentalTime

sealed class Gelu(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val SQRT2 = sqrt(2.0)

        fun gelu(array: MutableNumberNDArray): NumberNDArray {
            when (val type = array.type) {
                DataType.FLOAT -> {
                    array as MutableFloatNDArray
                    val pointer = array.array.pointer()
                    pointer.map(array.linearSize) {
                        0.5f * it * (1.0f + array.erfFor(it / SQRT2.toFloat()))
                    }
                }
                DataType.DOUBLE -> {
                    array as MutableDoubleNDArray
                    val pointer = array.array.pointer()
                    pointer.map(array.linearSize) {
                        0.5 * it * (1.0 + array.erfFor(it / SQRT2))
                    }
                }
                else -> throw IllegalStateException("Unsupported data type: $type")
            }
            return array
        }

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in GeluVer1.VERSION.asRange() -> GeluVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of Gelu operator: $version")
        }
    }
}

@ExperimentalTime
class GeluVer1(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Gelu(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "y", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("Gelu", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")
    }


    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs[0]!!.data as NumberNDArray
        return listOf(gelu(input.toMutable()).asTensor("Y"))
    }
}
