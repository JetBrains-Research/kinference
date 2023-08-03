package io.kinference.core.operators.reduce

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.reduce.l1.reduceL1
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.toIntArray

sealed class ReduceL1(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 18)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ReduceL1Ver1.VERSION.asRange() -> ReduceL1Ver1(name, attributes, inputs, outputs)
                in ReduceL1Ver18.VERSION.asRange() -> ReduceL1Ver18(name, attributes, inputs, outputs)
                else -> error("Unsupported version of ReduceL1 operator: $version")
            }
    }
}


class ReduceL1Ver1(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>, outputs: List<String>
) : ReduceL1(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
        ) + FLOAT_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false),
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "reduced", optional = false)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo(name = "axes", types = setOf(AttributeProto.AttributeType.INTS), required = false,  default = LongArray(0)),
            AttributeInfo(name = "keepdims", types = setOf(AttributeProto.AttributeType.INT), required = false, default = 1),
        )

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 18)
        private val INFO = OperatorInfo("ReduceL1", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axes: IntArray by attribute() { it: LongArray -> it.toIntArray() }
    private val keepDims: Boolean by attribute("keepdims") { it: Number -> it.toInt() == 1 }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs.first()!!.data as NumberNDArrayCore
        val actualAxes = if (axes.isEmpty()) input.shape.indices.toIntArray() else axes
        return listOf(input.reduceL1(actualAxes, keepDims).asTensor("reduced"))
    }
}

class ReduceL1Ver18(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>, outputs: List<String>
) : ReduceL1(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = NUMBER_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "axes", optional = true),
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "reduced", optional = false)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo(name = "keepdims", types = setOf(AttributeProto.AttributeType.INT), required = false, default = 1),
            AttributeInfo(name = "noop_with_empty_axes", types = setOf(AttributeProto.AttributeType.INT), required = false, default = 0)
        )

        internal val VERSION = VersionInfo(sinceVersion = 18)
        private val INFO = OperatorInfo("ReduceL1", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val keepDims: Boolean by attribute("keepdims") { it: Number -> it.toInt() == 1 }
    private val noopWithEmptyAxes: Boolean by attribute("noop_with_empty_axes") { it: Number -> it.toInt() == 1 }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs.first()!!.data as NumberNDArrayCore
        val axes = (inputs.getOrNull(1)?.data as LongNDArray?)?.array?.toArray()

        if (noopWithEmptyAxes && axes.isNullOrEmpty()) return listOf(input.asTensor("reduced"))

        val actualAxes = if (axes.isNullOrEmpty()) {
            input.shape.indices.toIntArray()
        } else {
            axes!!.toIntArray()
        }

        return listOf(input.reduceL1(actualAxes, keepDims).asTensor("reduced"))
    }
}
