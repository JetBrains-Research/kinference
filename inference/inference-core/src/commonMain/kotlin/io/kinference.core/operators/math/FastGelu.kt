package io.kinference.core.operators.math

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.acceptRecursive
import io.kinference.ndarray.arrays.pointers.map
import io.kinference.core.operators.*
import io.kinference.primitives.types.DataType
import kotlin.time.ExperimentalTime

@ExperimentalTime
class FastGelu(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) :
    Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
    companion object {
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


    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs.first()!!
        val bias = inputs.getOrNull(1)

        val result = when (input.data.type) {
            DataType.FLOAT -> {
                val biasData = bias?.data as? FloatNDArray
                val result = input.data.toMutable() as MutableFloatNDArray
                val pointer = result.array.pointer()
                if (biasData == null) {
                    pointer.map(result.linearSize) { fgelu(it) }
                } else {
                    pointer.acceptRecursive(biasData.array.pointer(), result.linearSize) { dst, src -> fgelu(dst + src) }
                }
                result
            }

            DataType.DOUBLE -> {
                val biasData = bias?.data as? DoubleNDArray
                val result = input.data.toMutable() as MutableDoubleNDArray
                val pointer = result.array.pointer()
                if (biasData == null) {
                    pointer.map(result.linearSize) { fgelu(it) }
                } else {
                    pointer.acceptRecursive(biasData.array.pointer(), result.linearSize) { dst, src -> fgelu(dst + src) }
                }
                result
            }

            else -> error("Unsupported operation")
        }.asTensor("Y")

        return listOf(result)
    }
}
