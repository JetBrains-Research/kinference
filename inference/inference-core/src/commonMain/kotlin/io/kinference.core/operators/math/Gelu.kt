package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.map
import io.kinference.ndarray.extensions.erf
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import kotlin.math.sqrt

sealed class Gelu(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val SQRT2 = sqrt(2.0)

        fun gelu(array: MutableNumberNDArrayCore): NumberNDArrayCore {
            when (val type = array.type) {
                DataType.FLOAT -> {
                    array as MutableFloatNDArray
                    val pointer = array.array.pointer()
                    pointer.map(array.linearSize) {
                        0.5f * it * (1.0f + erf(it / SQRT2.toFloat()))
                    }
                }
                DataType.DOUBLE -> {
                    array as MutableDoubleNDArray
                    val pointer = array.array.pointer()
                    pointer.map(array.linearSize) {
                        0.5 * it * (1.0 + erf(it / SQRT2))
                    }
                }
                else -> throw IllegalStateException("Unsupported data type: $type")
            }
            return array
        }

        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in GeluVer1.VERSION.asRange() -> GeluVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Gelu operator: $version")
        }
    }
}


class GeluVer1(name: String, attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Gelu(name, INFO, attributes, inputs, outputs) {
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


    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data as NumberNDArrayCore
        return listOf(gelu(input.toMutable()).asTensor("Y"))
    }
}
