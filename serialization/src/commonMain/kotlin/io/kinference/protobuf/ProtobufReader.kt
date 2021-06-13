package io.kinference.protobuf

import com.squareup.wire.*
import okio.BufferedSource
import okio.ByteString

class ProtobufReader(private val reader: ProtoReader) {
    constructor(source: BufferedSource) : this(ProtoReader(source))

    private data class ReaderState(var tag: Int = -1, var move: Boolean = true)

    private var state = ReaderState()

    private fun moveNext() {
        if (state.move) {
            state.tag = reader.nextTag()
        } else {
            state.move = true
        }
    }

    internal fun updateState(tag: Int, move: Boolean) {
        state.move = move
        state.tag = tag
    }

    internal fun nextTag() = reader.nextTag()

    internal fun forEachTag(handler: (Int) -> Unit): ByteString {
        val token = reader.beginMessage()
        while (true) {
            moveNext()
            if (state.tag == -1) break
            handler(state.tag)
        }
        return reader.endMessageAndGetUnknownFields(token)
    }

    fun addUnknownField(tag: Int, fieldEncoding: FieldEncoding, value: Any?) = reader.addUnknownField(tag, fieldEncoding, value)

    fun readUnknownField(tag: Int) = reader.readUnknownField(tag)
    fun <T> readValue(adapter: ProtoAdapter<T>) = adapter.decode(reader)

    fun readLong() = ProtoAdapter.INT64.decode(reader)
    fun readInt() = ProtoAdapter.INT32.decode(reader)
    fun readDouble() = ProtoAdapter.DOUBLE.decode(reader)
    fun readFloat() = ProtoAdapter.FLOAT.decode(reader)
    fun readULong() = ProtoAdapter.UINT64.decode(reader)

    fun readString() = ProtoAdapter.STRING.decode(reader)
    fun readBytes() = ProtoAdapter.BYTES.decode(reader)
}
