package io.kinference.tfjs.operators.seq

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.graph.Contexts
import io.kinference.ndarray.applyIf
import io.kinference.ndarray.arrays.isScalar
import io.kinference.ndarray.extensions.dataInt
import io.kinference.ndarray.extensions.tidyNDArrays
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.*
import io.kinference.types.ValueTypeInfo.SequenceTypeInfo

sealed class SplitToSequence(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>, outputs: List<String>
) : Operator<TFJSTensor, TFJSSequence>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(
            name: String,
            version: Int?,
            attributes: Map<String, Attribute<Any>>,
            inputs: List<String>,
            outputs: List<String>
        ): SplitToSequence {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SplitToSequenceVer11.VERSION.asRange() -> SplitToSequenceVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of SplitToSequence operator: $version")
            }
        }
    }
}


class SplitToSequenceVer11(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>, outputs: List<String>
) : SplitToSequence(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 0L),
            AttributeInfo("keepdims", setOf(AttributeProto.AttributeType.INT), false, default = 1L)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false),
            IOInfo(1, setOf(TensorProto.DataType.INT64, TensorProto.DataType.INT32), "split", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output_sequence", optional = false, onnxDataType = ONNXDataType.ONNX_SEQUENCE))

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("SplitToSequence", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }
    private val keepDims: Boolean by attribute("keepdims") { it: Number -> it.toInt() == 1 }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSSequence?> {
        val input = inputs[0]!!.data
        val parts = inputs.elementAtOrNull(1)?.data

        val tensors = tidyNDArrays {
            if (parts == null) {
                input.split(input.shape[axis], axis).applyIf(!keepDims) { segments ->
                    val newShape = IntArray(input.shape.size - 1)
                    input.shape.copyInto(newShape, 0, 0, axis)
                    input.shape.copyInto(newShape, axis, axis + 1)
                    segments.map { it.reshape(newShape) }
                }
            } else {
                val partsArray = parts.tfjsArray.dataInt()
                if (parts.isScalar()) input.split(partsArray[0], axis) else input.split(partsArray, axis)
            }.toTypedArray()
        }.map { it.asTensor() }

        return listOf(TFJSSequence("output_sequence", tensors, SequenceTypeInfo(tensors[0].info)))
    }
}
