package io.kinference.protobuf.arrays

import io.kinference.protobuf.ProtobufReader


internal class LongArrayBuilder(private var data: LongArray) : PrimitiveArrayBuilder<LongArray>() {
    override var position: Int = data.size
        private set

    init {
        checkCapacity(INITIAL_CAPACITY)
    }

    override fun checkCapacity(requiredCapacity: Int) {
        if (data.size < requiredCapacity)
            data = data.copyOf(requiredCapacity.coerceAtLeast(data.size * 2))
    }

    fun append(element: Long) {
        checkCapacity()
        data[position++] = element
    }

    override fun build() = data.copyOf(position)
}

internal object LongArrayDeserializer : PrimitiveArraySerializer<LongArray, LongArrayBuilder>() {
    override fun LongArray.toBuilder(): LongArrayBuilder = LongArrayBuilder(this)
    override fun empty(): LongArray = LongArray(0)

    override fun readElement(reader: ProtobufReader, builder: LongArrayBuilder) {
        builder.append(reader.readLong())
    }
}

internal object ULongArrayDeserializer : PrimitiveArraySerializer<LongArray, LongArrayBuilder>() {
    override fun LongArray.toBuilder(): LongArrayBuilder = LongArrayBuilder(this)
    override fun empty(): LongArray = LongArray(0)

    override fun readElement(reader: ProtobufReader, builder: LongArrayBuilder) {
        builder.append(reader.readULong())
    }
}
