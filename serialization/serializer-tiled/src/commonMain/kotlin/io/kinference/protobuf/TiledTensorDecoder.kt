package io.kinference.protobuf

import io.kinference.ndarray.arrays.tiled.*
import io.kinference.utils.InlineInt
import io.kinference.ndarray.extensions.createTiledArray
import io.kinference.protobuf.arrays.ArrayContainer
import io.kinference.protobuf.arrays.TiledArrayContainer
import io.kinference.protobuf.message.TensorProto

object TiledTensorDecoder : TensorDecoder() {
    override fun initContainer(): ArrayContainer = TiledArrayContainer()
    override suspend fun hasIntArray(proto: TensorProto): Boolean {
        return proto.getArrayData() is IntTiledArray
    }

    override suspend fun parseInt32Data(proto: TensorProto): Any {
        val data = proto.getArrayData() as IntTiledArray
        val pointer = data.pointer()

        return when (val type = proto.dataType) {
            TensorProto.DataType.BOOL -> BooleanTiledArray(proto.dims) { pointer.getAndIncrement() != 0 }
            TensorProto.DataType.INT8 -> ByteTiledArray(proto.dims) { pointer.getAndIncrement().toByte() }
            TensorProto.DataType.UINT8 -> UByteTiledArray(proto.dims) { pointer.getAndIncrement().toUByte() }
            TensorProto.DataType.INT16 -> ShortTiledArray(proto.dims) { pointer.getAndIncrement().toShort() }
            TensorProto.DataType.UINT16 -> UShortTiledArray(proto.dims) { pointer.getAndIncrement().toUShort() }
            TensorProto.DataType.BFLOAT16 -> FloatTiledArray(proto.dims) { pointer.getAndIncrement().parseAsBFloat() }
            TensorProto.DataType.FLOAT16 -> FloatTiledArray(proto.dims) { pointer.getAndIncrement().parseAsFloat16() }
            else -> error("Conversion from int32 to $type is not supported")
        }
    }


    override suspend fun makeArray(type: TensorProto.DataType, shape: IntArray, init: (InlineInt) -> Any): Any {
        return createTiledArray(type.resolveLocalDataType(), shape, init)
    }
}
