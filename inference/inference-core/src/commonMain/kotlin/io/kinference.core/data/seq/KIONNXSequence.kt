package io.kinference.core.data.seq

import io.kinference.core.CoreBackend
import io.kinference.core.KIONNXData
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.tensor.KITensor
import io.kinference.data.ONNXSequence
import io.kinference.protobuf.message.SequenceProto
import io.kinference.types.*

class KIONNXSequence(name: String?, data: List<KIONNXData<*>>, val info: ValueTypeInfo.SequenceTypeInfo) : ONNXSequence<List<KIONNXData<*>>, CoreBackend>(name, data) {
    constructor(name: String?, info: ValueTypeInfo.SequenceTypeInfo, size: Int, init: (Int) -> KIONNXData<*>) : this(name, List(size, init), info)
    constructor(data: List<KIONNXData<*>>, info: ValueInfo) : this(info.name, data, info.typeInfo as ValueTypeInfo.SequenceTypeInfo)

    override val backend = CoreBackend

    override fun close() {
        data.forEach { it.close() }
    }

    override fun clone(newName: String?): KIONNXSequence {
        return KIONNXSequence(newName, data.map { it.clone() }, info)
    }

    override fun rename(name: String): KIONNXSequence = KIONNXSequence(name, data, info)

    val length: Int = data.size

    companion object {
        fun create(proto: SequenceProto): KIONNXSequence {
            val elementTypeInfo = proto.extractTypeInfo()
            val name = proto.name!!
            val data = when (proto.elementType) {
                SequenceProto.DataType.TENSOR -> proto.tensorValues.map { KITensor.create(it) }
                SequenceProto.DataType.SEQUENCE -> proto.sequenceValues.map { create(it) }
                SequenceProto.DataType.MAP -> proto.mapValues.map { KIONNXMap.create(it) }
                else -> error("Unsupported sequence element type: ${proto.elementType}")
            }
            return KIONNXSequence(name, data, ValueTypeInfo.SequenceTypeInfo(elementTypeInfo))
        }

        internal fun SequenceProto.extractTypeInfo(): ValueTypeInfo = when (this.elementType) {
            SequenceProto.DataType.TENSOR -> {
                this.tensorValues.getOrNull(0)?.let {
                    ValueTypeInfo.TensorTypeInfo(TensorShape(it.dims), it.dataType!!)
                } ?: ValueTypeInfo.TensorTypeInfo()
            }
            SequenceProto.DataType.SEQUENCE -> {
                this.sequenceValues.getOrNull(0)?.let {
                    ValueTypeInfo.SequenceTypeInfo(it.extractTypeInfo())
                } ?: ValueTypeInfo.SequenceTypeInfo()
            }
            SequenceProto.DataType.MAP -> {
                this.mapValues.getOrNull(0)?.let {
                    val valueType = it.values!!.extractTypeInfo()
                    ValueTypeInfo.MapTypeInfo(keyType = it.keyType, valueType = valueType)
                } ?: ValueTypeInfo.MapTypeInfo()
            }
            else -> error("Unsupported sequence element type: ${this.elementType}")
        }
    }
}
