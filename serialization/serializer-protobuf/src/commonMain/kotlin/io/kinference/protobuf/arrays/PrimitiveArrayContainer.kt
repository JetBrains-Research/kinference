package io.kinference.protobuf.arrays

import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.PrimitiveArraySerializer.Companion.arraySerializer
import io.kinference.protobuf.message.TensorProto

internal class PrimitiveArrayContainer: ArrayContainer {
    private var _array: Any? = null
    override val array: Any?
        get() = _array

    override fun hasData() = _array != null

    override fun setData(newArray: Any) {
        _array = newArray
    }

    override suspend fun get(shape: IntArray): Any? = array

    override suspend fun decode(reader: ProtobufReader, tag: Int, dataType: TensorProto.DataType, shape: IntArray?) {
        _array = dataType.arraySerializer().decode(reader, tag)
    }
}
