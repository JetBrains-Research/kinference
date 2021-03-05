package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class TensorShapeProto(val dim: List<Dimension> = emptyList()) {
    companion object {
        fun decode(reader: ProtobufReader): TensorShapeProto {
            val dim = mutableListOf<Dimension>()
            reader.forEachTag { tag ->
                if (tag != 1) error("Unexpected tag $tag")
                dim.add(Dimension.decode(reader))
            }
            return TensorShapeProto(dim)
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
                    when (ReaderTag.fromInt(tag)) {
                        ReaderTag.DIM_VALUE -> dimValue = reader.readLong()
                        ReaderTag.DIM_PARAM -> dimParam = reader.readString()
                        ReaderTag.DENOTATION -> denotation = reader.readString()
                    }
                }
                return Dimension(denotation, dimValue, dimParam)
            }
        }

        private enum class ReaderTag(val tag: Int) {
            DIM_VALUE(1),
            DIM_PARAM(2),
            DENOTATION(3);

            companion object {
                fun fromInt(tag: Int) = values().first { it.tag == tag }
            }
        }
    }
}
