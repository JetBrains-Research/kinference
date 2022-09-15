package io.kinference.core.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.operator.Operator
import io.kinference.operator.OperatorInfo
import io.kinference.primitives.types.DataType
import kotlin.time.ExperimentalTime

@ExperimentalTime
abstract class Activation protected constructor(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {

    open fun activate(input: KITensor, contexts: Contexts<KIONNXData<*>>): KITensor = activate(input.data, contexts).asTensor()
    abstract fun activate(input: NDArrayCore, contexts: Contexts<KIONNXData<*>>): NDArrayCore

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        return listOf(activate(inputs.first()!!, contexts as Contexts<KIONNXData<*>>))
    }

    companion object {
        // TODO: Add activations with alpha and beta
        fun createFloat(name: String): FloatMap = when (name) {
            "Sigmoid" -> Sigmoid.activateFloat
            "Tanh" -> Tanh.activateFloat
            "Relu" -> Relu.activateFloat
            "Log" -> Log.activateFloat
            else -> throw UnsupportedOperationException()
        }

        fun createDouble(name: String): DoubleMap = when (name) {
            "Sigmoid" -> Sigmoid.activateDouble
            "Tanh" -> Tanh.activateDouble
            "Relu" -> Relu.activateDouble
            "Log" -> Log.activateDouble
            else -> throw UnsupportedOperationException()
        }

        fun create(name: String, type: DataType): PrimitiveToPrimitiveFunction = when (type) {
            DataType.DOUBLE -> createDouble(name)
            DataType.FLOAT -> createFloat(name)
            else -> throw UnsupportedOperationException()
        }
    }
}
