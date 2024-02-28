package io.kinference.protobuf

import io.kinference.protobuf.arrays.ArrayContainer
import io.kinference.protobuf.arrays.PrimitiveArrayContainer
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.InlineInt

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
            TensorProto.DataType.BFLOAT16 -> FloatArray(size) { data[it].parseAsBFloat() }
            TensorProto.DataType.FLOAT16 -> FloatArray(size) { data[it].parseAsFloat16() }
            else -> error("Conversion from int32 to $type is not supported")
        }
    }

    override fun makeArray(type: TensorProto.DataType, shape: IntArray, init: (InlineInt) -> Any): Any {
        val size = shape.fold(1, Int::times)
        return when (type) {
            TensorProto.DataType.DOUBLE -> DoubleArray(size) { init(InlineInt(it)) as Double }
            TensorProto.DataType.FLOAT -> FloatArray(size) { init(InlineInt(it)) as Float }
            TensorProto.DataType.FLOAT16 -> FloatArray(size) { init(InlineInt(it)) as Float }
            TensorProto.DataType.BFLOAT16 -> FloatArray(size) { init(InlineInt(it)) as Float }
            TensorProto.DataType.INT8 -> ByteArray(size) { init(InlineInt(it)) as Byte }
            TensorProto.DataType.INT16 -> ShortArray(size) { init(InlineInt(it)) as Short }
            TensorProto.DataType.INT32 -> IntArray(size) { init(InlineInt(it)) as Int }
            TensorProto.DataType.INT64 -> LongArray(size) { init(InlineInt(it)) as Long }
            TensorProto.DataType.UINT8 -> UByteArray(size) { init(InlineInt(it)) as UByte }
            TensorProto.DataType.UINT16 -> UShortArray(size) { init(InlineInt(it)) as UShort }
            TensorProto.DataType.UINT32 -> UIntArray(size) { init(InlineInt(it)) as UInt }
            TensorProto.DataType.UINT64 -> ULongArray(size) { init(InlineInt(it)) as ULong }
            TensorProto.DataType.BOOL -> BooleanArray(size) { init(InlineInt(it)) as Boolean }
            else -> error("Unsupported data type: $type")
        }
    }
}
