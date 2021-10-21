package io.kinference.core.operators.math

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.core.operators.*
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.ndarray.arrays.NumberNDArray
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
class TopK(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
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
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "K", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Values", optional = false),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "Indices", optional = false)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, -1),
            AttributeInfo("largest", setOf(AttributeProto.AttributeType.INT), false, 1),
            AttributeInfo("sorted", setOf(AttributeProto.AttributeType.INT), false, 1),
        )

        private val INFO = OperatorInfo("ReduceSum", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val axis: Int by attribute() { it: Long -> it.toInt() }
    private val largest: Boolean by attribute() { it: Number -> it.toInt() == 1 }
    private val sorted: Boolean by attribute() { it: Number -> it.toInt() == 1 }

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs.first()!!.data as NumberNDArray
        val k = (inputs[1]!!.data as LongNDArray).singleValue().toInt()

        val (values, indices) = input.topK(axis, k, largest, sorted)

        return listOf(values.asTensor("Values"), indices.asTensor("Indices"))
    }
}

