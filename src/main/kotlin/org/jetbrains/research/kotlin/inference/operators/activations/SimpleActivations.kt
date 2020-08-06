package org.jetbrains.research.kotlin.inference.operators.activations

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.extensions.functional.DoubleArrayToDoubleArray
import org.jetbrains.research.kotlin.inference.extensions.functional.FloatArrayToFloatArray
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.IOInfo
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo
import kotlin.math.exp
import kotlin.math.max

class Identity(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INFO = OperatorInfo("Identity", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = input
}

class Relu(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Relu", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )

        fun activationFloat() = FloatArrayToFloatArray { array ->
            for (i in array.indices) array[i] = max(0.0f, array[i])
            array
        }

        fun activationDouble() = DoubleArrayToDoubleArray { array ->
            for (i in array.indices) array[i] = max(0.0, array[i])
            array
        }
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = when (input.type) {
        TensorProto.DataType.FLOAT -> input.mapElements(activationFloat())
        TensorProto.DataType.DOUBLE -> input.mapElements(activationDouble())
        else -> error("Unsupported operation")
    }
}

class Sigmoid(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Sigmoid", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )

        fun activationFloat() = FloatArrayToFloatArray { array ->
            for (i in array.indices) array[i] = 1.0f / (1.0f + exp(-array[i]))
            array
        }

        fun activationDouble() = DoubleArrayToDoubleArray { array ->
            for (i in array.indices) array[i] = 1.0 / (1.0 + exp(-array[i]))
            array
        }
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = when (input.type) {
        TensorProto.DataType.FLOAT -> input.mapElements(activationFloat())
        TensorProto.DataType.DOUBLE -> input.mapElements(activationDouble())
        else -> error("Unsupported operation")
    }
}

fun fexp(x: Double): Double {
    val tmp = (1512775 * x + 1072632447).toLong()
    return java.lang.Double.longBitsToDouble(tmp shl 32)
}

class Tanh(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Tanh", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )

        fun activationFloat() = FloatArrayToFloatArray { array ->
            for (i in array.indices) {
                var temp = exp(2.0f * array[i])
                if (temp.isInfinite()) temp = Float.MAX_VALUE
                array[i] = ((temp - 1.0f) / (temp + 1.0f))
            }
            array
        }

        fun activationDouble() = DoubleArrayToDoubleArray { array ->
            for (i in array.indices) array[i] = ((exp(2.0 * array[i]) - 1.0) / (exp(2.0 * array[i]) + 1.0))
            array
        }
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = when (input.type) {
        TensorProto.DataType.FLOAT -> input.mapElements(activationFloat())
        TensorProto.DataType.DOUBLE -> input.mapElements(activationDouble())
        else -> error("Unsupported operation")
    }
}
