package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.ndarray.extensions.reshape
import io.kinference.operators.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
class Pad(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES - TensorProto.DataType.BOOL

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = false),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "pads", optional = false, differentiable = false),
            IOInfo(2, TYPE_CONSTRAINTS, "constant_value", optional = true)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("mode", setOf(AttributeProto.AttributeType.STRING), required = false, default = "constant")
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false, differentiable = false))

        private val INFO = OperatorInfo("Pad", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val mode: String by attribute()

    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        val input = inputs[0]!!.data
        val pads = inputs[1]!!.data as LongNDArray
        val padsData = pads.array.toArray()
        val constantValue = inputs.getOrNull(2)?.data

        val padsNormalized = Array(input.rank) { padsData[it].toInt() to padsData[it + input.rank].toInt() }

        val output = input.pad(padsNormalized, mode, constantValue)
        return listOf(output.asTensor("output"))
    }
}
