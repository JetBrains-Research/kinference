package io.kinference.tfjs.data.seq

import io.kinference.data.ONNXSequence
import io.kinference.protobuf.message.SequenceProto
import io.kinference.tfjs.TFJSBackend
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.types.*

class TFJSSequence(name: String?, data: List<TFJSData<*>>, val info: ValueTypeInfo.SequenceTypeInfo) : ONNXSequence<List<TFJSData<*>>, TFJSBackend>(name, data) {
    constructor(name: String?, info: ValueTypeInfo.SequenceTypeInfo, size: Int, init: (Int) -> TFJSData<*>) : this(name, List(size, init), info)
    constructor(data: List<TFJSData<*>>, info: ValueInfo) : this(info.name, data, info.typeInfo as ValueTypeInfo.SequenceTypeInfo)

    override val backend = TFJSBackend

    override fun rename(name: String) = TFJSSequence(name, data, info)

    override suspend fun close() {
        data.forEach { it.close() }
    }

    override suspend fun clone(newName: String?): TFJSSequence {
        return TFJSSequence(newName, data.map { it.clone() }, info)
    }

    val length: Int = data.size

    companion object {
        suspend fun create(proto: SequenceProto): TFJSSequence {
            val elementTypeInfo = proto.extractTypeInfo()
            val name = proto.name!!
            val data = when (proto.elementType) {
                SequenceProto.DataType.TENSOR -> proto.tensorValues.map { TFJSTensor.create(it) }
                SequenceProto.DataType.SEQUENCE -> proto.sequenceValues.map { create(it) }
                SequenceProto.DataType.MAP -> proto.mapValues.map { TFJSMap.create(it) }
                else -> error("Unsupported sequence element type: ${proto.elementType}")
            }
            return TFJSSequence(name, data, ValueTypeInfo.SequenceTypeInfo(elementTypeInfo))
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
