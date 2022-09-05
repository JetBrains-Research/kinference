package io.kinference.protobuf.arrays

import io.kinference.protobuf.ProtobufReader

abstract class ArrayBuilder<Array> {
    abstract fun build(): Array
}

abstract class ArraySerializer<Array, Builder : ArrayBuilder<Array>> {
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
