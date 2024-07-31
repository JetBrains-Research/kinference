package io.kinference.protobuf.arrays

import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.message.TensorProto


internal abstract class TiledArraySerializer<Array, Builder : ArrayBuilder<Array>> : ArraySerializer<Array, Builder>() {
    protected abstract suspend fun empty(shape: IntArray): Array

    suspend fun decode(reader: ProtobufReader, shape: IntArray, initialTag: Int): Array {
        val builder = empty(shape).toBuilder()
        doRead(reader, builder, initialTag)
        return builder.build()
    }

    companion object {
        fun TensorProto.DataType.tiledSerializer() = when (this) {
            TensorProto.DataType.DOUBLE -> DoubleTiledArraySerializer
            TensorProto.DataType.FLOAT -> FloatTiledArraySerializer
            TensorProto.DataType.INT32 -> IntTiledArraySerializer
            TensorProto.DataType.INT64 -> LongTiledArraySerializer
            TensorProto.DataType.UINT64 -> ULongTiledArraySerializer
            else -> error("Deserializer for $this not found")
        }
    }
}
