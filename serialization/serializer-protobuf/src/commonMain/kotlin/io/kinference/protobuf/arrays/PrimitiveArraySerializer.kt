package io.kinference.protobuf.arrays

import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.message.TensorProto

abstract class PrimitiveArrayBuilder<Array> : ArrayBuilder<Array>() {
    protected abstract val position: Int
    internal abstract fun checkCapacity(requiredCapacity: Int = position + 1)

    companion object {
        const val INITIAL_CAPACITY = 50
    }
}

abstract class PrimitiveArraySerializer<Array, Builder : PrimitiveArrayBuilder<Array>> : ArraySerializer<Array, Builder>() {
    protected abstract fun empty(): Array

    fun decode(reader: ProtobufReader, initialTag: Int): Array {
        val builder = empty().toBuilder()
        doRead(reader, builder, initialTag)
        return builder.build()
    }

    companion object {
        fun TensorProto.DataType.arraySerializer() = when (this) {
            TensorProto.DataType.DOUBLE -> DoubleArraySerializer
            TensorProto.DataType.FLOAT -> FloatArraySerializer
            TensorProto.DataType.INT32 -> IntArraySerializer
            TensorProto.DataType.INT64 -> LongArraySerializer
            TensorProto.DataType.UINT64 -> ULongArraySerializer
            else -> error("Deserializer for $this not found")
        }
    }
}
