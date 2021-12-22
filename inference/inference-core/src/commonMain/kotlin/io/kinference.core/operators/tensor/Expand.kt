package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIContext
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.operator.*
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.ndarray.arrays.pointers.forEachIndexed
import io.kinference.ndarray.arrays.tiled.LongTiledArray
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

sealed class Expand(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 8)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ExpandVer8.VERSION.asRange() -> ExpandVer8(attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

@OptIn(ExperimentalTime::class)
class ExpandVer8(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Expand(INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "input", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false, differentiable = true))

        internal val VERSION = VersionInfo(sinceVersion = 8)
        private val INFO = OperatorInfo("Expand", emptySet(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    internal fun LongTiledArray.toIntArray(): IntArray {
        val output = IntArray(this.size)
        this.pointer().forEachIndexed(this.size) { index, value -> output[index] = value.toInt() }

        return output
    }

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<KITensor?>, profilingContext: ProfilingContext?, checkCancelled: () -> Unit): List<KITensor?> {
        val input = inputs[0]!!.data
        val shapeNDArray = inputs[1]!!.data as LongNDArray

        val shape = shapeNDArray.array.toIntArray()
        return listOf(input.expand(shape).asTensor("output"))

    }

}

