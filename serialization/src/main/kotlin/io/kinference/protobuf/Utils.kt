package io.kinference.protobuf

import io.kinference.primitives.types.DataType
import io.kinference.protobuf.arrays.*
import io.kinference.protobuf.message.TensorProto


fun ProtobufReader.readLongArray(tag: Int) = LongArraySerializer.decode(this, tag)
fun ProtobufReader.readULongArray(tag: Int) = ULongArraySerializer.decode(this, tag)
fun ProtobufReader.readFloatArray(tag: Int) = FloatArraySerializer.decode(this, tag)

internal fun ProtobufReader.readFloatTiledArray(
    tag: Int,
    dims: IntArray?,
    dest: TiledArrayContainer? = null
) = (dest ?: TiledArrayContainer()).decode(this, tag, TensorProto.DataType.FLOAT, dims)

internal fun ProtobufReader.readDoubleTiledArray(
    tag: Int,
    dims: IntArray?,
    dest: TiledArrayContainer? = null
) = (dest ?: TiledArrayContainer()).decode(this, tag, TensorProto.DataType.DOUBLE, dims)

internal fun ProtobufReader.readIntTiledArray(
    tag: Int,
    dims: IntArray?,
    dest: TiledArrayContainer? = null
) = (dest ?: TiledArrayContainer()).decode(this, tag, TensorProto.DataType.INT32, dims)

internal fun ProtobufReader.readLongTiledArray(
    tag: Int,
    dims: IntArray?,
    dest: TiledArrayContainer? = null
) = (dest ?: TiledArrayContainer()).decode(this, tag, TensorProto.DataType.INT64, dims)

internal fun ProtobufReader.readULongTiledArray(
    tag: Int,
    dims: IntArray?,
    dest: TiledArrayContainer? = null
) = (dest ?: TiledArrayContainer()).decode(this, tag, TensorProto.DataType.UINT64, dims)

fun TensorProto.DataType.resolveLocalDataType(): DataType {
    return when (this) {
        TensorProto.DataType.DOUBLE -> DataType.DOUBLE
        TensorProto.DataType.FLOAT, TensorProto.DataType.FLOAT16 -> DataType.FLOAT
        TensorProto.DataType.INT32 -> DataType.INT
        TensorProto.DataType.INT64 -> DataType.LONG
        TensorProto.DataType.INT16 -> DataType.SHORT
        TensorProto.DataType.INT8 -> DataType.BYTE
        TensorProto.DataType.BOOL -> DataType.BOOLEAN
        TensorProto.DataType.UINT32 -> DataType.UINT
        TensorProto.DataType.UINT64 -> DataType.ULONG
        TensorProto.DataType.UINT16 -> DataType.USHORT
        TensorProto.DataType.UINT8 -> DataType.UBYTE
        else -> error("Cannot resolve data type")
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
        else -> error("Cannot resolve data type")
    }
}
