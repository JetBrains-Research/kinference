package org.jetbrains.research.kotlin.inference.operators.activations

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.TypedNDArray
import org.jetbrains.research.kotlin.inference.extensions.functional.DoubleArrayToDoubleArray
import org.jetbrains.research.kotlin.inference.extensions.functional.FloatArrayToFloatArray
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.IOInfo
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo
import org.jetbrains.research.kotlin.inference.operators.math.tanh
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

    override fun activate(input: TypedNDArray<Any>): TypedNDArray<Any> = input
}

class Relu(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Relu", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )

        val activateFloat = FloatArrayToFloatArray { array ->
            for (i in array.indices) array[i] = max(0.0f, array[i])
            array
        }

        val activateDouble = DoubleArrayToDoubleArray { array ->
            for (i in array.indices) array[i] = max(0.0, array[i])
            array
        }
    }

    override fun activate(input: TypedNDArray<Any>): TypedNDArray<Any> = when (input.type) {
        TensorProto.DataType.FLOAT -> input.mapElements(activateFloat)
        TensorProto.DataType.DOUBLE -> input.mapElements(activateDouble)
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

        val activateFloat = FloatArrayToFloatArray { array ->
            for (i in array.indices) array[i] = 1.0f / (1.0f + exp(-array[i]))
            array
        }

        val activateDouble = DoubleArrayToDoubleArray { array ->
            for (i in array.indices) array[i] = 1.0 / (1.0 + exp(-array[i]))
            array
        }
    }

    override fun activate(input: TypedNDArray<Any>): TypedNDArray<Any> = when (input.type) {
        TensorProto.DataType.FLOAT -> input.mapElements(activateFloat)
        TensorProto.DataType.DOUBLE -> input.mapElements(activateDouble)
        else -> error("Unsupported operation")
    }
}

class Tanh(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Tanh", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )

        val activateFloat = FloatArrayToFloatArray { array ->
            for (i in array.indices) array[i] = tanh(array[i])
            array
        }

        val activateDouble = DoubleArrayToDoubleArray { array ->
            for (i in array.indices) array[i] = tanh(array[i])
            array
        }
    }

    override fun activate(input: TypedNDArray<Any>): TypedNDArray<Any> = when (input.type) {
        TensorProto.DataType.FLOAT -> input.mapElements(activateFloat)
        TensorProto.DataType.DOUBLE -> input.mapElements(activateDouble)
        else -> error("Unsupported operation")
    }
}
