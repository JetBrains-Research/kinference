package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.LongArraySerializer
import io.kinference.protobuf.readTensor

class SparseTensorProto(
    val values: TensorProto? = null,
    val indices: TensorProto? = null,
    val dims: LongArray? = null
) {
    companion object {
        fun decode(reader: ProtobufReader): SparseTensorProto {
            var values: TensorProto? = null
            var indices: TensorProto? = null
            var dims: LongArray? = null
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.VALUES -> values = reader.readTensor()
                    ReaderTag.INDICES -> indices = reader.readTensor()
                    ReaderTag.DIMS -> dims = LongArraySerializer.decode(reader, tag)
                    null -> reader.readUnknownField(tag)
                }
            }
            return SparseTensorProto(values = values, indices = indices, dims = dims)
        }
    }

    private enum class ReaderTag(val tag: Int) {
        VALUES(1),
        INDICES(2),
        DIMS(3);

        companion object {
            fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
        }
    }
}
