package io.kinference.protobuf

import io.kinference.primitives.types.DataType
import io.kinference.protobuf.arrays.*
import io.kinference.protobuf.message.TensorProto

fun LongArray.toIntArray() = IntArray(size) { this[it].toInt() }
fun IntArray.toLongArray() = LongArray(size) { this[it].toLong() }

fun ProtobufReader.readFloatArray(tag: Int) = FloatArraySerializer.decode(this, tag)
fun ProtobufReader.readDoubleArray(tag: Int) = DoubleArraySerializer.decode(this, tag)
fun ProtobufReader.readIntArray(tag: Int) = IntArraySerializer.decode(this, tag)
fun ProtobufReader.readLongArray(tag: Int) = LongArraySerializer.decode(this, tag)
fun ProtobufReader.readULongArray(tag: Int) = ULongArraySerializer.decode(this, tag)

internal fun ProtobufReader.readFloatArray(tag: Int, dims: IntArray?, dest: ArrayContainer) = dest.decode(this, tag, TensorProto.DataType.FLOAT, dims)
internal fun ProtobufReader.readDoubleArray(tag: Int, dims: IntArray?, dest: ArrayContainer) = dest.decode(this, tag, TensorProto.DataType.DOUBLE, dims)
internal fun ProtobufReader.readIntArray(tag: Int, dims: IntArray?, dest: ArrayContainer) = dest.decode(this, tag, TensorProto.DataType.INT32, dims)
internal fun ProtobufReader.readLongArray(tag: Int, dims: IntArray?, dest: ArrayContainer) = dest.decode(this, tag, TensorProto.DataType.INT64, dims)
internal fun ProtobufReader.readULongArray(tag: Int, dims: IntArray?, dest: ArrayContainer) = dest.decode(this, tag, TensorProto.DataType.UINT64, dims)

fun ProtobufReader.readTensor() = config.tensorDecoder.decode(this)

val FLOAT_TENSOR_TYPES = setOf(TensorProto.DataType.FLOAT, TensorProto.DataType.FLOAT16, TensorProto.DataType.BFLOAT16)

fun TensorProto.DataType.resolveLocalDataType(): DataType {
    return when (this) {
        TensorProto.DataType.DOUBLE -> DataType.DOUBLE
        in FLOAT_TENSOR_TYPES -> DataType.FLOAT
        TensorProto.DataType.INT32 -> DataType.INT
        TensorProto.DataType.INT64 -> DataType.LONG
        TensorProto.DataType.INT16 -> DataType.SHORT
        TensorProto.DataType.INT8 -> DataType.BYTE
        TensorProto.DataType.BOOL -> DataType.BOOLEAN
        TensorProto.DataType.UINT32 -> DataType.UINT
        TensorProto.DataType.UINT64 -> DataType.ULONG
        TensorProto.DataType.UINT16 -> DataType.USHORT
        TensorProto.DataType.UINT8 -> DataType.UBYTE
        else -> error("Cannot resolve data type: $this")
    }
}

fun DataType.resolveProtoDataType(): TensorProto.DataType {
    return when (this) {
        DataType.DOUBLE -> TensorProto.DataType.DOUBLE
        DataType.FLOAT -> TensorProto.DataType.FLOAT
        DataType.INT -> TensorProto.DataType.INT32
        DataType.LONG -> TensorProto.DataType.INT64
        DataType.SHORT -> TensorProto.DataType.INT16
        DataType.BYTE -> TensorProto.DataType.INT8
        DataType.BOOLEAN -> TensorProto.DataType.BOOL
        DataType.UINT -> TensorProto.DataType.UINT32
        DataType.ULONG -> TensorProto.DataType.UINT64
        DataType.USHORT -> TensorProto.DataType.UINT16
        DataType.UBYTE -> TensorProto.DataType.UINT8
        else -> error("Cannot resolve data type: $this")
    }
}
