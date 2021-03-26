package io.kinference.operators.math

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.map
import io.kinference.operators.*
import io.kinference.primitives.types.DataType
import kotlin.math.sqrt
import kotlin.time.ExperimentalTime

@ExperimentalTime
class Gelu(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) :
    Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val SQRT2 = sqrt(2.0)

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "y", optional = false)
        )

        private val INFO = OperatorInfo("Gelu", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)


        fun gelu(array: MutableNumberNDArray): NumberNDArray {
            when (array.type) {
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
                else -> throw IllegalStateException("Unsupported type")
            }
            return array
        }
    }


    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        val input = inputs[0]!!.data as NumberNDArray
        return listOf(gelu(input.toMutable()).asTensor("Y"))
    }
}
