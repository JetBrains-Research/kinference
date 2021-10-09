package io.kinference.core.data.seq

import io.kinference.core.data.KIONNXData
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.types.*
import io.kinference.data.ONNXDataType
import io.kinference.protobuf.message.SequenceProto

class KIONNXSequence(data: List<KIONNXData<*>>, info: ValueInfo) : KIONNXData<List<KIONNXData<*>>>(ONNXDataType.ONNX_SEQUENCE, data, info) {
    constructor(info: ValueInfo, size: Int, init: (Int) -> KIONNXData<*>) : this(List(size, init), info)

    override fun rename(name: String): KIONNXSequence = KIONNXSequence(data, ValueInfo(info.typeInfo, name))

    val length: Int = data.size

    companion object {
        fun create(proto: SequenceProto): KIONNXSequence {
            val elementTypeInfo = proto.extractTypeInfo()
            val info = ValueInfo(name = proto.name!!, typeInfo = elementTypeInfo)
            val data = when (proto.elementType) {
                SequenceProto.DataType.TENSOR -> proto.tensorValues.map { KITensor.create(it) }
                SequenceProto.DataType.SEQUENCE -> proto.sequenceValues.map { create(it) }
                SequenceProto.DataType.MAP -> proto.mapValues.map { KIONNXMap.create(it) }
                else -> error("")
            }
            return KIONNXSequence(data, info)
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
