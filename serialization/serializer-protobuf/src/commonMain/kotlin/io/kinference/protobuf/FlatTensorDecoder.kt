package io.kinference.protobuf

import io.kinference.protobuf.arrays.ArrayContainer
import io.kinference.protobuf.arrays.PrimitiveArrayContainer
import io.kinference.protobuf.message.TensorProto

object FlatTensorDecoder : TensorDecoder() {
    override fun initContainer(): ArrayContainer = PrimitiveArrayContainer()
    override fun hasIntArray(proto: TensorProto): Boolean {
        return proto.arrayData is IntArray
    }

    override fun parseInt32Data(proto: TensorProto): Any {
        val data = proto.arrayData as IntArray
        val size = data.size

        return when (val type = proto.dataType) {
            TensorProto.DataType.BOOL -> BooleanArray(size) { data[it] != 0 }
            TensorProto.DataType.INT8 -> ByteArray(size) { data[it].toByte() }
            TensorProto.DataType.UINT8 -> UByteArray(size) { data[it].toUByte() }
            TensorProto.DataType.INT16 -> ShortArray(size) { data[it].toShort() }
            TensorProto.DataType.UINT16 -> UShortArray(size) { data[it].toUShort() }
            else -> error("Conversion from int32 to $type is not supported")
        }
    }

    override fun makeArray(type: TensorProto.DataType, shape: IntArray, init: (Int) -> Any): Any {
        val size = shape.fold(1, Int::times)
        return when (type) {
            TensorProto.DataType.DOUBLE -> DoubleArray(size) { init(it) as Double }
            TensorProto.DataType.FLOAT -> FloatArray(size) { init(it) as Float }
            TensorProto.DataType.INT64 -> LongArray(size) { init(it) as Long }
            TensorProto.DataType.INT32 -> IntArray(size) { init(it) as Int }
            TensorProto.DataType.INT16 -> ShortArray(size) { init(it) as Short }
            TensorProto.DataType.UINT16 -> UShortArray(size) { init(it) as UShort }
            TensorProto.DataType.BOOL -> BooleanArray(size) { init(it) as Boolean }
            TensorProto.DataType.INT8 -> ByteArray(size) { init(it) as Byte }
            TensorProto.DataType.UINT8 -> UByteArray(size) { init(it) as UByte }
            else -> error("Unsupported data type: $type")
        }
    }
}
