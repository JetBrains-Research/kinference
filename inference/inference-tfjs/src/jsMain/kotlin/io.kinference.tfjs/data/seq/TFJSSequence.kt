package io.kinference.tfjs.data.seq

import io.kinference.data.*
import io.kinference.protobuf.message.SequenceProto
import io.kinference.tfjs.TFJSBackend
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.types.*

class TFJSSequence(name: String?, data: List<TFJSData<*>>, val info: ValueTypeInfo.SequenceTypeInfo) : ONNXSequence<List<TFJSData<*>>, TFJSBackend>(name, data) {
    constructor(name: String?, info: ValueTypeInfo.SequenceTypeInfo, size: Int, init: (Int) -> TFJSData<*>) : this(name, List(size, init), info)
    constructor(data: List<TFJSData<*>>, info: ValueInfo) : this(info.name, data, info.typeInfo as ValueTypeInfo.SequenceTypeInfo)

    override val backend = TFJSBackend

    override fun rename(name: String) = TFJSSequence(name, data, info)

    val length: Int = data.size

    companion object {
        fun create(proto: SequenceProto): TFJSSequence {
            val elementTypeInfo = proto.extractTypeInfo() as ValueTypeInfo.SequenceTypeInfo
            val name = proto.name!!
            val data = when (proto.elementType) {
                SequenceProto.DataType.TENSOR -> proto.tensorValues.map { TFJSTensor.create(it) }
                SequenceProto.DataType.SEQUENCE -> proto.sequenceValues.map { create(it) }
                SequenceProto.DataType.MAP -> proto.mapValues.map { TFJSMap.create(it) }
                else -> error("Unsupported sequence element type: ${proto.elementType}")
            }
            return TFJSSequence(name, data, elementTypeInfo)
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
            else -> error("Unsupported sequence element type: ${this.elementType}")
        }
    }
}
