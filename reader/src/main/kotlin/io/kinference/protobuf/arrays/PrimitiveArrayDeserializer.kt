package io.kinference.protobuf.arrays

import io.kinference.protobuf.ProtobufReader

internal abstract class PrimitiveArrayBuilder<Array> {
    protected abstract val position: Int
    internal abstract fun checkCapacity(requiredCapacity: Int = position + 1)
    internal abstract fun build(): Array

    companion object {
        const val INITIAL_CAPACITY = 50
    }
}

internal abstract class PrimitiveArraySerializer<Array, ArrayBuilder : PrimitiveArrayBuilder<Array>> {
    protected abstract fun empty(): Array
    abstract fun Array.toBuilder(): ArrayBuilder

    fun builder(): ArrayBuilder = empty().toBuilder()

    protected abstract fun readElement(reader: ProtobufReader, builder: ArrayBuilder)

    fun decode(reader: ProtobufReader, initialTag: Int): Array {
        val builder = builder()
        var currentTag: Int
        while (true) {
            readElement(reader, builder)
            currentTag = reader.nextTag()
            if (currentTag != initialTag) break
        }
        reader.updateState(tag = currentTag, move = false)
        return builder.build()
    }
}
