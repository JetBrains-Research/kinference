package io.kinference.core.data.seq

import io.kinference.core.CoreBackend
import io.kinference.core.KIONNXData
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.types.*
import io.kinference.data.*
import io.kinference.protobuf.message.SequenceProto

class KIONNXSequence(name: String?, data: List<KIONNXData<*>>, val info: ValueTypeInfo.SequenceTypeInfo) : ONNXSequence<List<KIONNXData<*>>, CoreBackend>(name, data) {
    constructor(name: String?, info: ValueTypeInfo.SequenceTypeInfo, size: Int, init: (Int) -> KIONNXData<*>) : this(name, List(size, init), info)
    constructor(data: List<KIONNXData<*>>, info: ValueInfo) : this(info.name, data, info.typeInfo as ValueTypeInfo.SequenceTypeInfo)

    override val backend = CoreBackend

    override fun rename(name: String): KIONNXSequence = KIONNXSequence(name, data, info)

    val length: Int = data.size

    companion object {
        fun create(proto: SequenceProto): KIONNXSequence {
            val elementTypeInfo = proto.extractTypeInfo() as ValueTypeInfo.SequenceTypeInfo
            val name = proto.name!!
            val data = when (proto.elementType) {
                SequenceProto.DataType.TENSOR -> proto.tensorValues.map { KITensor.create(it) }
                SequenceProto.DataType.SEQUENCE -> proto.sequenceValues.map { create(it) }
                SequenceProto.DataType.MAP -> proto.mapValues.map { KIONNXMap.create(it) }
                else -> error("")
            }
            return KIONNXSequence(name, data, elementTypeInfo)
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
