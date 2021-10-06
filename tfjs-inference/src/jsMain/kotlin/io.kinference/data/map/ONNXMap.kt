package io.kinference.data.map

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.data.seq.ONNXSequence
import io.kinference.data.seq.ONNXSequence.Companion.extractTypeInfo
import io.kinference.protobuf.message.MapProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.types.ValueInfo
import io.kinference.types.ValueTypeInfo

class ONNXMap(val data: Map<Any, ONNXData>, info: ValueInfo) : ONNXData(ONNXDataType.ONNX_MAP, info) {
    val keyType: TensorProto.DataType
        get() = (info.typeInfo as ValueTypeInfo.MapTypeInfo).keyType

    val valueType: ValueTypeInfo
        get() = (info.typeInfo as ValueTypeInfo.MapTypeInfo).valueType

    override fun rename(name: String): ONNXData = ONNXMap(data, ValueInfo(info.typeInfo, name))

    companion object {
        fun create(proto: MapProto): ONNXMap {
            val elementType = ValueTypeInfo.MapTypeInfo(proto.keyType, proto.values!!.extractTypeInfo())
            val info = ValueInfo(elementType, proto.name!!)
            val map = HashMap<Any, ONNXData>().apply {
                val keys = if (proto.keyType == TensorProto.DataType.STRING) proto.stringKeys else castKeys(proto.keys!!, proto.keyType)
                keys.zip(ONNXSequence.create(proto.values!!).data) { key, value -> put(key, value) }
            }
            return ONNXMap(map, info)
        }

        private fun castKeys(keys: LongArray, type: TensorProto.DataType) = when (type) {
            TensorProto.DataType.UINT8 -> keys.map { it.toUByte() }
            TensorProto.DataType.INT8 -> keys.map { it.toByte() }
            TensorProto.DataType.UINT16 -> keys.map { it.toUShort() }
            TensorProto.DataType.INT16 -> keys.map { it.toShort() }
            TensorProto.DataType.INT32 -> keys.map { it.toInt() }
            TensorProto.DataType.INT64 -> keys.map { it.toLong() }
            TensorProto.DataType.UINT32 -> keys.map { it.toUInt() }
            TensorProto.DataType.UINT64 -> keys.map { it.toULong() }
            else -> error("")
        }
    }
}
