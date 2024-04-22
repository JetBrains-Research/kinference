package io.kinference.protobuf.arrays

import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.message.TensorProto

interface ArrayContainer {
    val array: Any?

    suspend fun decode(reader: ProtobufReader, tag: Int, dataType: TensorProto.DataType, shape: IntArray?)
    fun hasData(): Boolean
    fun setData(newArray: Any)
    suspend fun get(shape: IntArray): Any?
}

fun ArrayContainer?.hasData() = this != null && this.hasData()
