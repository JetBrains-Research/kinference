package io.kinference.protobuf

import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.createTiledArray
import io.kinference.protobuf.arrays.ArrayContainer
import io.kinference.protobuf.arrays.TiledArrayContainer
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.InlineInt

object TiledTensorDecoder : TensorDecoder() {
    override fun initContainer(): ArrayContainer = TiledArrayContainer()
    override fun hasIntArray(proto: TensorProto): Boolean {
        return proto.arrayData is IntTiledArray
    }

    override fun parseInt32Data(proto: TensorProto): Any {
        val data = proto.arrayData as IntTiledArray
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


    override fun makeArray(type: TensorProto.DataType, shape: IntArray, init: (InlineInt) -> Any): Any {
        return createTiledArray(type.resolveLocalDataType(), shape, init)
    }
}
