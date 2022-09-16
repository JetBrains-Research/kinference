package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.*
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.ndarray.arrays.NDArrayCore
import io.kinference.ndarray.toIntArray
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

sealed class Tile(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when(version ?: DEFAULT_VERSION.sinceVersion) {
            in TileVer6.VERSION.asRange() -> TileVer6(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Tile operator: $version")
        }
    }
}

@ExperimentalTime
class TileVer6(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Tile(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "repeats", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false, differentiable = true)
        )

        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Tile", emptySet(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    @Suppress("UNCHECKED_CAST")
    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data
        val repeats = inputs[1]!!.data as LongNDArray
        val repeatsIntArray = repeats.array.toArray().toIntArray()

        val output = input.tile(repeatsIntArray) as NDArrayCore

        return listOf(output.asTensor("output"))
    }
}
