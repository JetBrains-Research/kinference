package io.kinference.core.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.ndarray.arrays.*
import io.kinference.operator.Operator
import io.kinference.operator.OperatorInfo
import io.kinference.primitives.types.DataType
import io.kinference.profiler.ProfilingContext
import kotlin.time.ExperimentalTime

@ExperimentalTime
abstract class Activation protected constructor(
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {

    open fun activate(input: KITensor): KITensor = this.activate(input.data).asTensor()
    abstract fun activate(input: NDArray): NDArray

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<KITensor?>, profilingContext: ProfilingContext?, checkCancelled: () -> Unit): List<KITensor?> {
        return listOf(activate(inputs.first()!!))
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
