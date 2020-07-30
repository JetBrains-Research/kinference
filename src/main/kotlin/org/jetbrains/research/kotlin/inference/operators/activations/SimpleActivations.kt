package org.jetbrains.research.kotlin.inference.operators.activations

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.InputInfo
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo
import org.jetbrains.research.kotlin.inference.operators.OutputInfo
import kotlin.math.exp
import kotlin.math.max

class Identity(attributes: Map<String, Attribute<Any>> = emptyMap(), usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INFO = OperatorInfo("Identity", emptyMap(),
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = input
}

class Relu(attributes: Map<String, Attribute<Any>> = emptyMap(), usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Relu", emptyMap(),
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )

        inline fun activationFloat() = object : FloatArrayToFloatArray {
            override fun apply(array: FloatArray): FloatArray {
                for (i in array.indices) array[i] = max(0.0f, array[i])
                return array
            }
        }

        inline fun activationDouble() = object : DoubleArrayToDoubleArray {
            override fun apply(array: DoubleArray): DoubleArray {
                for (i in array.indices) array[i] = max(0.0, array[i])
                return array
            }
        }
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = when (input.type) {
        TensorProto.DataType.FLOAT -> input.mapElements(activationFloat())
        TensorProto.DataType.DOUBLE -> input.mapElements(activationDouble())
        else -> error("Unsupported operation")
    }
}

class Sigmoid(attributes: Map<String, Attribute<Any>> = emptyMap(), usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Sigmoid", emptyMap(),
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )

        inline fun activationFloat() = object : FloatArrayToFloatArray {
            override fun apply(array: FloatArray): FloatArray {
                for (i in array.indices) array[i] = 1.0f / (1.0f + exp(-array[i]))
                return array
            }
        }

        inline fun activationDouble() = object : DoubleArrayToDoubleArray {
            override fun apply(array: DoubleArray): DoubleArray {
                for (i in array.indices) array[i] = 1.0 / (1.0 + exp(-array[i]))
                return array
            }
        }
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = when (input.type) {
        TensorProto.DataType.FLOAT -> input.mapElements(activationFloat())
        TensorProto.DataType.DOUBLE -> input.mapElements(activationDouble())
        else -> error("Unsupported operation")
    }
}

class Tanh(attributes: Map<String, Attribute<Any>> = emptyMap(), usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Tanh", emptyMap(),
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )

        fun activationFloat() = object : FloatArrayToFloatArray {
            override fun apply(array: FloatArray): FloatArray {
                for (i in array.indices) array[i] = ((exp(2.0 * array[i]) - 1.0) / (exp(2.0 * array[i]) + 1.0)).toFloat()
                return array
            }
        }

        fun activationDouble() = object : DoubleArrayToDoubleArray {
            override fun apply(array: DoubleArray): DoubleArray {
                for (i in array.indices) array[i] = ((exp(2.0 * array[i]) - 1.0) / (exp(2.0 * array[i]) + 1.0))
                return array
            }
        }
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = when (input.type) {
        TensorProto.DataType.FLOAT -> input.mapElements(activationFloat())
        TensorProto.DataType.DOUBLE -> input.mapElements(activationDouble())
        else -> error("Unsupported operation")
    }
}
