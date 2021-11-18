package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIContext
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.acceptRecursive
import io.kinference.ndarray.arrays.pointers.map
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import kotlin.time.ExperimentalTime

sealed class FastGelu(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in FastGeluVer1.VERSION.asRange() -> FastGeluVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of FastGelu operator: $version")
        }
    }
}

@ExperimentalTime
class FastGeluVer1(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : FastGelu(INFO, attributes, inputs, outputs) {
    companion object {
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

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs.first()!!
        val bias = inputs.getOrNull(1)

        val result = when (val type = input.data.type) {
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

            else -> error("Unsupported operation for data type $type")
        }.asTensor("Y")

        return listOf(result)
    }
}
