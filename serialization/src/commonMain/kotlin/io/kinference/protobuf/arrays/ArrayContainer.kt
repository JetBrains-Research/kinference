package io.kinference.protobuf.arrays

import io.kinference.ndarray.extensions.createArray
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.PrimitiveArraySerializer.Companion.arraySerializer
import io.kinference.protobuf.arrays.TiledArraySerializer.Companion.tiledSerializer
import io.kinference.protobuf.message.TensorProto

enum class ArrayFormat {
    PRIMITIVE, TILED, OTHER;

    internal fun container() = when (this) {
        PRIMITIVE -> PrimitiveArrayContainer()
        TILED -> TiledArrayContainer()
        OTHER -> null
    }
}

interface ArrayContainer {
    val array: Any?

    fun decode(reader: ProtobufReader, tag: Int, dataType: TensorProto.DataType, shape: IntArray?)
    fun hasData(): Boolean
    fun setData(newArray: Any)
    fun get(shape: IntArray): Any?
}

fun ArrayContainer?.hasData() = this != null && this.hasData()

internal class PrimitiveArrayContainer : ArrayContainer {
    private var _array: Any? = null
    override val array: Any?
        get() = _array

    override fun hasData() = _array != null

    override fun setData(newArray: Any) {
        _array = newArray
    }

    override fun get(shape: IntArray): Any? = array

    override fun decode(reader: ProtobufReader, tag: Int, dataType: TensorProto.DataType, shape: IntArray?) {
        _array = dataType.arraySerializer().decode(reader, tag)
    }
}

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

    override fun decode(reader: ProtobufReader, tag: Int, dataType: TensorProto.DataType, shape: IntArray?) {
        if (shape != null) {
            tiled = dataType.tiledSerializer().decode(reader, shape, tag)
            tiledInitialized = true
        } else {
            _array = dataType.arraySerializer().decode(reader, tag)
        }
    }

    override fun get(shape: IntArray): Any? {
        if (!hasData()) return null

        if (!tiledInitialized) {
            tiled = createArray(shape, array!!)
            _array = null
            tiledInitialized = true
        }
        return tiled!!
    }
}
