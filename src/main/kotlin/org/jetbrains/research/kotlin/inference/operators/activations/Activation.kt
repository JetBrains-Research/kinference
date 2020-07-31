package org.jetbrains.research.kotlin.inference.operators.activations

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.functional.*
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
        fun createFloat(name: String): FloatArrayToFloatArray = when (name) {
            "Sigmoid" -> Sigmoid.activationFloat()
            "Tanh" -> Tanh.activationFloat()
            "Relu" -> Relu.activationFloat()
            else -> throw UnsupportedOperationException()
        }

        fun createDouble(name: String): DoubleArrayToDoubleArray = when (name) {
            "Sigmoid" -> Sigmoid.activationDouble()
            "Tanh" -> Tanh.activationDouble()
            "Relu" -> Relu.activationDouble()
            else -> throw UnsupportedOperationException()
        }

        fun create(name: String, type: TensorProto.DataType): PrimitiveArrayFunction = when (type) {
            TensorProto.DataType.DOUBLE -> createDouble(name)
            TensorProto.DataType.FLOAT -> createFloat(name)
            else -> throw UnsupportedOperationException()
        }
    }
}
