package io.kinference.tfjs.data.seq

import io.kinference.tfjs.data.ONNXData
import io.kinference.tfjs.data.ONNXDataType
import io.kinference.tfjs.data.map.ONNXMap
import io.kinference.tfjs.data.tensors.Tensor
import io.kinference.protobuf.message.SequenceProto
import io.kinference.tfjs.types.*

class ONNXSequence(val data: List<ONNXData>, info: ValueInfo) : ONNXData(ONNXDataType.ONNX_SEQUENCE, info) {
    constructor(info: ValueInfo, size: Int, init: (Int) -> ONNXData) : this(List(size, init), info)

    override fun rename(name: String): ONNXData = ONNXSequence(data, ValueInfo(info.typeInfo, name))

    val length: Int = data.size

    companion object {
        fun create(proto: SequenceProto): ONNXSequence {
            val elementTypeInfo = proto.extractTypeInfo()
            val info = ValueInfo(name = proto.name!!, typeInfo = elementTypeInfo)
            val data = when (proto.elementType) {
                SequenceProto.DataType.TENSOR -> proto.tensorValues.map { Tensor.create(it) }
                SequenceProto.DataType.SEQUENCE -> proto.sequenceValues.map { create(it) }
                SequenceProto.DataType.MAP -> proto.mapValues.map { ONNXMap.create(it) }
                else -> error("")
            }
            return ONNXSequence(data, info)
        }

        internal fun SequenceProto.extractTypeInfo(): ValueTypeInfo = when (this.elementType) {
            SequenceProto.DataType.TENSOR -> {
                val first = this.tensorValues[0]
                ValueTypeInfo.TensorTypeInfo(TensorShape(first.dims), first.dataType!!)
            }
            SequenceProto.DataType.SEQUENCE -> ValueTypeInfo.SequenceTypeInfo(this.sequenceValues[0].extractTypeInfo())
            SequenceProto.DataType.MAP -> {
                val first = this.mapValues[0]
                val valueType = first.values!!.extractTypeInfo()
                ValueTypeInfo.MapTypeInfo(keyType = first.keyType, valueType = valueType)
            }
            else -> error("")
        }
    }
}
