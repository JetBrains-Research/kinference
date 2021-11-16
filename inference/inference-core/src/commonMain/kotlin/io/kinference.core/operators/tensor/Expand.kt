package io.kinference.core.operators.tensor

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.core.operators.*
import io.kinference.core.operators.VersionInfo.Companion.asRange
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.ndarray.arrays.pointers.forEachIndexed
import io.kinference.ndarray.arrays.tiled.LongTiledArray
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class Expand(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "input", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false, differentiable = true))

        private val VERSION = VersionInfo(sinceVersion = 8)
        private val INFO = OperatorInfo("Expand", emptySet(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in VERSION.asRange() -> Expand(attributes, inputs, outputs)
            else -> error("Unsupported version of Expand operator: $version")
        }
    }

    internal fun LongTiledArray.toIntArray(): IntArray {
        val output = IntArray(this.size)
        this.pointer().forEachIndexed(this.size) { index, value -> output[index] = value.toInt() }

        return output
    }

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs[0]!!.data
        val shapeNDArray = inputs[1]!!.data as LongNDArray

        val shape = shapeNDArray.array.toIntArray()
        return listOf(input.expand(shape).asTensor("output"))

    }

}

