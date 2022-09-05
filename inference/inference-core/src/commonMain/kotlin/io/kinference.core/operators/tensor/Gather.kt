package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.indexAxis
import io.kinference.ndarray.extensions.gather
import io.kinference.operator.*
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

sealed class Gather(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in GatherVer1.VERSION.asRange() -> GatherVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

@ExperimentalTime
class GatherVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Gather(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT32, TensorProto.DataType.INT64), "indices", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("Gather", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }


    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val (data, indices) = inputs
        val axis = data!!.data.indexAxis(axis)
        return listOf(data.data.gather(indices!!.data, axis).asTensor())
    }
}
