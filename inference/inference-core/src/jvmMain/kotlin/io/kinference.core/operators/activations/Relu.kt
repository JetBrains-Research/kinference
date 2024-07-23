package io.kinference.core.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.constants.ByteConstants
import io.kinference.ndarray.extensions.constants.ShortConstants
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import kotlin.math.max

sealed class Relu(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Activation(name, info, attributes, inputs, outputs) {
    companion object {
        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = max(0.0f, value)
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = max(0.0, value)
        }

        val activateInt = object : IntMap {
            override fun apply(value: Int): Int = max(0, value)
        }

        val activateLong = object : LongMap {
            override fun apply(value: Long ): Long  = max(0L, value)
        }

        val activateByte = object : ByteMap {
            override fun apply(value: Byte): Byte = maxOf(ByteConstants.ZERO, value)
        }

        val activateShort = object : ShortMap {
            override fun apply(value: Short): Short = maxOf(ShortConstants.ZERO, value)
        }

        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Relu {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ReluVer6.VERSION.asRange() -> ReluVer6(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Relu operator: $version")
            }
        }
    }
}


class ReluVer6(
    name: String,
    attributes: Map<String, Attribute<Any>> = emptyMap(),
    inputs: List<String>,
    outputs: List<String>
) : Relu(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES + INT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Relu", emptyMap(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun activate(input: NDArrayCore, contexts: Contexts<KIONNXData<*>>): NDArrayCore {
        return when (val type = input.type) {
            DataType.FLOAT -> input.map(activateFloat)
            DataType.DOUBLE -> input.map(activateDouble)
            DataType.INT -> input.map(activateInt)
            DataType.LONG -> input.map(activateLong)
            DataType.BYTE -> input.map(activateByte)
            DataType.SHORT -> input.map(activateShort)
            else -> error("Unsupported data type for this operation: $type")
        }
    }
}
