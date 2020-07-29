package org.jetbrains.research.kotlin.inference.operators.activations

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.Operator
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo

@Suppress("UNCHECKED_CAST")
abstract class Activation(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int)
    : Operator<Tensor, Tensor>(info, usedOutputsNum, attributes) {

    open fun activate(input: Tensor): Tensor = this.activate(input.data).asTensor()
    abstract fun activate(input: NDArray<Any>): NDArray<Any>

    override fun apply(inputs: List<Tensor>): List<Tensor> {
        return listOf(activate(inputs.first()))
    }

    companion object {
        // TODO: Add activations with alpha and beta
        fun createFloat(name: String): (Float) -> Float = when (name) {
            "Sigmoid" -> { i -> Sigmoid.activateFloat(i) }
            "Tanh" -> { i -> Tanh.activateFloat(i) }
            "Relu" -> { i -> Relu.activateFloat(i) }
            else -> throw UnsupportedOperationException()
        }

        fun createDouble(name: String): (Double) -> Double = when (name) {
            "Sigmoid" -> { i -> Sigmoid.activateDouble(i) }
            "Tanh" -> { i -> Tanh.activateDouble(i) }
            "Relu" -> { i -> Relu.activateDouble(i) }
            else -> throw UnsupportedOperationException()
        }

        fun create(name: String, type: TensorProto.DataType): (Any) -> Any = when (type) {
            TensorProto.DataType.DOUBLE -> createDouble(name)
            TensorProto.DataType.FLOAT -> createFloat(name)
            else -> throw UnsupportedOperationException()
        } as (Any) -> Any
    }
}
