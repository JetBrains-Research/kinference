package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIContext
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.extensions.indexAxis
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import kotlin.time.ExperimentalTime

sealed class Flatten(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in FlattenVer1.VERSION.asRange() -> FlattenVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

@OptIn(ExperimentalTime::class)
class FlattenVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Flatten(INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "input", optional = false, differentiable = true),
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false, differentiable = true))

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), default = 1, required = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("Flatten", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    val axis: Int by attribute() { it: Number -> it.toInt() }

    private fun makeShape(shape: IntArray, axis: Int): IntArray {
        val firstDimension = shape.slice(0 until axis).fold(1, Int::times)
        val secondDimension = shape.slice(axis until shape.size).fold(1, Int::times)

        return intArrayOf(firstDimension, secondDimension)
    }

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<KITensor?>, profilingContext: ProfilingContext?, checkCancelled: () -> Unit): List<KITensor?> {
        val input = inputs[0]!!.data
        val actualAxis = input.indexAxis(axis)

        val newShape = makeShape(input.shape, actualAxis)
        return listOf(input.reshape(newShape).asTensor("output"))
    }

}
