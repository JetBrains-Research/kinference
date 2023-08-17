package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.arrays.indexAxis
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import kotlin.math.max
import kotlin.math.min

sealed class Shape(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Shape {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ShapeVer1.VERSION.asRange() -> ShapeVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Shape operator: $version")
            }
        }
    }
}

class ShapeVer1 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Shape(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false))

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("start", setOf(AttributeProto.AttributeType.INT), default = 0L, required = false),
            AttributeInfo("end", setOf(AttributeProto.AttributeType.INT), default = null, required = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("Shape", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val start by attribute { it: Number -> it.toInt() }
    private val end by attributeOrNull { it: Number? -> it?.toInt() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val output = tidyNDArray {
            val input = inputs[0]!!.data
            val shape = input.shape

            val actualStart = max(0, input.indexAxis(start))
            val actualEnd = min(shape.size, input.indexAxis(end ?: shape.size))
            val outputShape = shape.sliceArray(actualStart until actualEnd)

            return@tidyNDArray NDArrayTFJS.int(outputShape, arrayOf(outputShape.size))
        }

        return listOf(output.asTensor("shape"))
    }
}
