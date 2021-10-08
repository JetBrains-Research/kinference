package io.kinference.tfjs.data.seq

import io.kinference.data.ONNXDataType
import io.kinference.protobuf.message.SequenceProto
import io.kinference.tfjs.data.TFJSData
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.types.*

class TFJSSequence(data: List<TFJSData<*>>, info: ValueInfo) : TFJSData<List<TFJSData<*>>>(data, info) {
    override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
    constructor(info: ValueInfo, size: Int, init: (Int) -> TFJSData<*>) : this(List(size, init), info)

    override fun rename(name: String) = TFJSSequence(data, ValueInfo(info.typeInfo, name))

    val length: Int = data.size

    companion object {
        fun create(proto: SequenceProto): TFJSSequence {
            val elementTypeInfo = proto.extractTypeInfo()
            val info = ValueInfo(name = proto.name!!, typeInfo = elementTypeInfo)
            val data = when (proto.elementType) {
                SequenceProto.DataType.TENSOR -> proto.tensorValues.map { TFJSTensor.create(it) }
                SequenceProto.DataType.SEQUENCE -> proto.sequenceValues.map { create(it) }
                SequenceProto.DataType.MAP -> proto.mapValues.map { TFJSMap.create(it) }
                else -> error("")
            }
            return TFJSSequence(data, info)
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
