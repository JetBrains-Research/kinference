package io.kinference.tfjs.operators.reduce

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.dataInt
import io.kinference.ndarray.extensions.reduceMean
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.utils.toIntArray
import io.kinference.utils.toTypedIntArray

sealed class ReduceMean(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>, outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): ReduceMean {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ReduceMeanVer1.VERSION.asRange() -> ReduceMeanVer1(name, attributes, inputs, outputs)
                in ReduceMeanVer18.VERSION.asRange() -> ReduceMeanVer18(name, attributes, inputs, outputs)
                else -> error("Unsupported version of ReduceMean operator: $version")
            }
        }
    }
}


class ReduceMeanVer1(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : ReduceMean(name, INFO, attributes, inputs, outputs) {
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
            AttributeInfo("axes", setOf(AttributeProto.AttributeType.INTS), false, LongArray(0)),
            AttributeInfo("keepdims", setOf(AttributeProto.AttributeType.INT), false, 1),
        )

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 18)
        private val INFO = OperatorInfo("ReduceMean", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axes: Array<Int> by attribute() { array: LongArray -> Array(array.size) { array[it].toInt() } }
    private val keepDims: Boolean by attribute("keepdims") { it: Number -> it.toInt() == 1 }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data as NumberNDArrayTFJS
        val actualAxes = if (axes.isEmpty()) input.shape.indices.toTypedIntArray() else axes
        return listOf(input.reduceMean(actualAxes, keepDims).asTensor("reduced"))
    }
}

class ReduceMeanVer18(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>, outputs: List<String>
) : ReduceMean(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
        ) + FLOAT_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "axes", optional = true),
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "reduced", optional = false)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("keepdims", setOf(AttributeProto.AttributeType.INT), false, 1),
            AttributeInfo("noop_with_empty_axes", setOf(AttributeProto.AttributeType.INT), false, 0)
        )

        internal val VERSION = VersionInfo(sinceVersion = 18)
        private val INFO = OperatorInfo("ReduceMean", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val keepDims: Boolean by attribute("keepdims") { it: Number -> it.toInt() == 1 }
    private val noopWithEmptyAxes: Boolean by attribute("noop_with_empty_axes") { it: Number -> it.toInt() == 1 }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs.first()!!.data as NumberNDArrayTFJS
        val axes = (inputs.getOrNull(1)?.data as NumberNDArrayTFJS?)?.dataInt()

        if (noopWithEmptyAxes && axes.isNullOrEmpty()) return listOf(input.asTensor("reduced"))

        val actualAxes = if (axes.isNullOrEmpty()) {
            input.shape.indices.toTypedIntArray()
        } else {
            axes!!.toTypedArray()
        }

        return listOf(input.reduceMean(actualAxes, keepDims).asTensor("reduced"))
    }
}
