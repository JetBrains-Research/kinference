package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class TensorShapeProto(val dim: List<Dimension> = emptyList()) {
    companion object {
        fun decode(reader: ProtobufReader): TensorShapeProto {
            val dim = mutableListOf<Dimension>()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> dim.add(Dimension.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return TensorShapeProto(dim = dim)
        }
    }

    data class Dimension(
        val denotation: String? = null,
        val dimValue: Long? = null,
        val dimParam: String? = null
    ) {
        companion object {
            fun decode(reader: ProtobufReader): Dimension {
                var denotation: String? = null
                var dimValue: Long? = null
                var dimParam: String? = null
                reader.forEachTag { tag ->
                    when (tag) {
                        1 -> dimValue = reader.readLong()
                        2 -> dimParam = reader.readString()
                        3 -> denotation = reader.readString()
                        else -> reader.readUnknownField(tag)
                    }
                }
                return Dimension(denotation = denotation, dimValue = dimValue, dimParam = dimParam)
            }
        }
    }
}
