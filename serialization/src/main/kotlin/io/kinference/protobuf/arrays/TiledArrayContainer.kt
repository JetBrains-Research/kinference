package io.kinference.protobuf.arrays

import io.kinference.ndarray.extensions.createArray
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.PrimitiveArraySerializer.Companion.arraySerializer
import io.kinference.protobuf.arrays.TiledArraySerializer.Companion.tiledSerializer
import io.kinference.protobuf.message.TensorProto

internal class TiledArrayContainer {
    private var array: Any? = null
    private var tiled: Any? = null

    private var tiledInitialized: Boolean = false

    fun hasData() = array != null || tiled != null

    fun setTiled(array: Any) {
        tiled = array
        tiledInitialized = true
    }

    fun decode(reader: ProtobufReader, tag: Int, dataType: TensorProto.DataType, shape: IntArray?) {
        if (shape != null) {
            tiled = dataType.tiledSerializer().decode(reader, shape, tag)
            tiledInitialized = true
        } else {
            array = dataType.arraySerializer().decode(reader, tag)
        }
    }

    fun get(shape: IntArray): Any? {
        if (!hasData()) return null

        if (!tiledInitialized) {
            tiled = createArray(shape, array!!)
            array = null
            tiledInitialized = true
        }
        return tiled!!
    }
}
