package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import kotlin.math.exp

class Identity(attributes: Map<String, Attribute<Any>> = emptyMap()) : Activation("Identity", TYPE_CONSTRAINTS, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT8,
            TensorProto.DataType.UINT16,
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT8,
            TensorProto.DataType.INT16,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.BFLOAT16,
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.STRING,
            TensorProto.DataType.BOOL,
            TensorProto.DataType.COMPLEX64,
            TensorProto.DataType.COMPLEX128
        )
    }

    override fun activate(input: Tensor): Tensor = input
}

class Relu(attributes: Map<String, Attribute<Any>> = emptyMap()) : Activation("Relu", TYPE_CONSTRAINTS, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.BFLOAT16
        )
    }

    override fun activate(input: Tensor): Tensor = input.mapElements { x -> max(0, x as Number) }
}

class Sigmoid(attributes: Map<String, Attribute<Any>> = emptyMap()) : Activation("Sigmoid", TYPE_CONSTRAINTS, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.BFLOAT16
        )
    }

    override fun activate(input: Tensor): Tensor = input.mapElements { x ->
        1.0 / (1.0 + exp(-(x as Number).toDouble()))
    }
}

class Tanh(attributes: Map<String, Attribute<Any>> = emptyMap()) : Activation("Tanh", TYPE_CONSTRAINTS, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.BFLOAT16
        )
    }

    override fun activate(input: Tensor): Tensor = input.mapElements { x ->
        x as Number
        (exp(2.0 * x.toDouble()) - 1.0) / (exp(2.0 * x.toDouble()) + 1.0)
    }
}
