package io.kinference.protobuf.arrays

import io.kinference.ndarray.extensions.tiledFromPrimitiveArray
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.PrimitiveArraySerializer.Companion.arraySerializer
import io.kinference.protobuf.arrays.TiledArraySerializer.Companion.tiledSerializer
import io.kinference.protobuf.message.TensorProto

internal class TiledArrayContainer : ArrayContainer {
    private var _array: Any? = null
    private var tiled: Any? = null
    override val array: Any?
        get() = tiled

    private var tiledInitialized: Boolean = false

    override fun hasData() = array != null || tiled != null

    override fun setData(newArray: Any) {
        tiled = newArray
        tiledInitialized = true
    }

    override suspend fun decode(reader: ProtobufReader, tag: Int, dataType: TensorProto.DataType, shape: IntArray?) {
        if (shape != null) {
            tiled = dataType.tiledSerializer().decode(reader, shape, tag)
            tiledInitialized = true
        } else {
            _array = dataType.arraySerializer().decode(reader, tag)
        }
    }

    override suspend fun get(shape: IntArray): Any? {
        if (!hasData()) return null

        if (!tiledInitialized) {
            tiled = tiledFromPrimitiveArray(shape, array!!)
            _array = null
            tiledInitialized = true
        }
        return tiled!!
    }
}
