package org.jetbrains.research.kotlin.inference.operators.activations

import org.jetbrains.research.kotlin.inference.annotations.DataType
import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.data.tensors.asTensor
import org.jetbrains.research.kotlin.inference.graph.Context
import org.jetbrains.research.kotlin.inference.ndarray.*
import org.jetbrains.research.kotlin.inference.operators.Operator
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo

abstract class Activation(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(info, attributes, inputs, outputs) {

    open fun activate(input: Tensor): Tensor = this.activate(input.data).asTensor()
    abstract fun activate(input: NDArray): NDArray

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        return listOf(activate(inputs.first()!!))
    }

    @ExperimentalUnsignedTypes
    companion object {
        // TODO: Add activations with alpha and beta
        fun createFloat(name: String): FloatMap = when (name) {
            "Sigmoid" -> Sigmoid.activateFloat
            "Tanh" -> Tanh.activateFloat
            "Relu" -> Relu.activateFloat
            else -> throw UnsupportedOperationException()
        }

        fun createDouble(name: String): DoubleMap = when (name) {
            "Sigmoid" -> Sigmoid.activateDouble
            "Tanh" -> Tanh.activateDouble
            "Relu" -> Relu.activateDouble
            else -> throw UnsupportedOperationException()
        }

        fun create(name: String, type: DataType): PrimitiveToPrimitiveFunction = when (type) {
            DataType.DOUBLE -> createDouble(name)
            DataType.FLOAT -> createFloat(name)
            else -> throw UnsupportedOperationException()
        }
    }
}
