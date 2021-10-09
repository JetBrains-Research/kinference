package io.kinference.data

interface ONNXDataAdapter<SourceType, TargetType : ONNXData<*>> {
    fun toONNXData(data: SourceType): TargetType
    fun fromONNXData(data: TargetType): SourceType

    companion object {
        fun <T : ONNXData<*>> idAdapter() : ONNXDataAdapter<T, T> = object : ONNXDataAdapter<T, T> {
            override fun toONNXData(data: T): T = data
            override fun fromONNXData(data: T): T = data
        }
    }
}
