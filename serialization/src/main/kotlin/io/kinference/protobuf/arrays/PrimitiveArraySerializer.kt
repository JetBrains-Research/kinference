package io.kinference.protobuf.arrays

import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.message.TensorProto


internal abstract class ArrayBuilder<Array> {
    internal abstract fun build(): Array
}

internal abstract class PrimitiveArrayBuilder<Array> : ArrayBuilder<Array>() {
    protected abstract val position: Int
    internal abstract fun checkCapacity(requiredCapacity: Int = position + 1)

    companion object {
        const val INITIAL_CAPACITY = 50
    }
}

internal abstract class ArraySerializer<Array, Builder : ArrayBuilder<Array>> {
    abstract fun Array.toBuilder(): Builder

    protected abstract fun readElement(reader: ProtobufReader, builder: Builder)

    protected fun doRead(reader: ProtobufReader, builder: Builder, initialTag: Int) {
        var currentTag: Int
        while (true) {
            readElement(reader, builder)
            currentTag = reader.nextTag()
            if (currentTag != initialTag) break
        }
        reader.updateState(tag = currentTag, move = false)
    }
}

internal abstract class PrimitiveArraySerializer<Array, Builder : PrimitiveArrayBuilder<Array>> : ArraySerializer<Array, Builder>() {
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

internal abstract class TiledArraySerializer<Array, Builder : ArrayBuilder<Array>> : ArraySerializer<Array, Builder>() {
    protected abstract fun empty(shape: IntArray): Array

    fun decode(reader: ProtobufReader, shape: IntArray, initialTag: Int): Array {
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
