package io.kinference.core.operators.seq

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.*
import io.kinference.data.*
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.types.TensorShape
import io.kinference.types.ValueTypeInfo

sealed class SequenceInsert(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KIONNXData<*>, KIONNXSequence>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): SequenceInsert {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SequenceInsertVer11.VERSION.asRange() -> SequenceInsertVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of SequenceInsert operator: $version")
            }
        }
    }
}


class SequenceInsertVer11(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : SequenceInsert(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input_sequence", optional = false, onnxDataType = ONNXDataType.ONNX_SEQUENCE),
            IOInfo(1, TYPE_CONSTRAINTS, "tensor", optional = false),
            IOInfo(2, setOf(TensorProto.DataType.INT64, TensorProto.DataType.INT32), "position", optional = true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output_sequence", optional = false, onnxDataType = ONNXDataType.ONNX_SEQUENCE)
        )

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("SequenceInsert", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KIONNXData<*>?>): List<KIONNXSequence?> {
        val seq = ArrayList(inputs[0]!!.data as List<KITensor>)
        val tensor = inputs[1]!! as KITensor
        val positionTensor = inputs.getOrNull(2)?.data as? NumberNDArrayCore

        val position = (positionTensor?.singleValue() as? Number)?.toInt() ?: seq.size
        val actualPosition = if (position >= 0) position else seq.size - position

        require(actualPosition >= 0 && actualPosition <= seq.size) { "Index $position is out of range [-${seq.size}, ${seq.size}]" }

        seq.add(actualPosition, tensor)
        val outputSeq = KIONNXSequence(
            name = "output_sequence",
            data = seq,
            info = ValueTypeInfo.SequenceTypeInfo(
                elementType = ValueTypeInfo.TensorTypeInfo(
                    shape = TensorShape.unknown(),
                    type = tensor.info.type
                )
            )
        )
        return listOf(outputSeq)
    }
}
