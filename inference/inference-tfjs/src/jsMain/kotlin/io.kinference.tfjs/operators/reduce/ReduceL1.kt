package io.kinference.tfjs.operators.reduce

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.dataInt
import io.kinference.ndarray.extensions.reduceL1
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.utils.toIntArray

sealed class ReduceL1(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
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
        private val TYPE_CONSTRAINTS = NUMBER_DATA_TYPES

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

    private val axes: IntArray by attribute { array: LongArray -> array.toIntArray() }
    private val keepDims: Boolean by attribute("keepdims") { it: Number -> it.toInt() == 1 }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs.first()!!.data as NumberNDArrayTFJS
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

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs.first()!!.data as NumberNDArrayTFJS
        val axes = (inputs.getOrNull(1)?.data as NumberNDArrayTFJS?)?.dataInt()

        if (noopWithEmptyAxes && axes.isNullOrEmpty()) return listOf(input.asTensor("reduced"))

        val actualAxes = if (axes.isNullOrEmpty()) {
            input.shape.indices.toIntArray()
        } else {
            axes!!
        }

        return listOf(input.reduceL1(actualAxes, keepDims).asTensor("reduced"))
    }
}
