package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.ndarray.arrays.pointers.forEachIndexed
import io.kinference.ndarray.arrays.tiled.LongTiledArray
import io.kinference.operators.*
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class Expand(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "input", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false, differentiable = true))

        private val INFO = OperatorInfo("Expand", emptySet(), INPUTS_INFO, OUTPUTS_INFO)
    }

    internal fun LongTiledArray.toIntArray(): IntArray {
        val output = IntArray(this.size)
        this.pointer().forEachIndexed(this.size) { index, value -> output[index] = value.toInt() }

        return output
    }

    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        val input = inputs[0]!!.data
        val shapeNDArray = inputs[1]!!.data as LongNDArray

        val shape = shapeNDArray.array.toIntArray()
        return listOf(input.expand(shape).asTensor("output"))

    }

}

