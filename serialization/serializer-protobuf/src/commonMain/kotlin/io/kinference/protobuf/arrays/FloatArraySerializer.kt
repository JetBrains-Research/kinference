package io.kinference.protobuf.arrays

import io.kinference.protobuf.ProtobufReader

internal class FloatArrayBuilder(private var data: FloatArray) : PrimitiveArrayBuilder<FloatArray>() {
    override var position: Int = data.size
        private set

    init {
        checkCapacity(INITIAL_CAPACITY)
    }

    override fun checkCapacity(requiredCapacity: Int) {
        if (data.size < requiredCapacity)
            data = data.copyOf(requiredCapacity.coerceAtLeast(data.size * 2))
    }

    fun append(element: Float) {
        checkCapacity()
        data[position++] = element
    }

    override fun build() = data.copyOf(position)
}

internal object FloatArraySerializer : PrimitiveArraySerializer<FloatArray, FloatArrayBuilder>() {
    override fun FloatArray.toBuilder(): FloatArrayBuilder = FloatArrayBuilder(this)
    override fun empty(): FloatArray = FloatArray(0)

    override fun readElement(reader: ProtobufReader, builder: FloatArrayBuilder) {
        builder.append(reader.readFloat())
    }
}
