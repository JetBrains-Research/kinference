package io.kinference.data

interface ONNXDataAdapter<SourceType, TargetType : ONNXData<*>> {
    fun toONNXData(name: String, data: SourceType): TargetType
    fun fromONNXData(data: TargetType): SourceType
}

object IdAdapter : ONNXDataAdapter<ONNXData<*>, ONNXData<*>>  {
    override fun toONNXData(name: String, data: ONNXData<*>): ONNXData<*> = data
    override fun fromONNXData(data: ONNXData<*>): ONNXData<*> = data
}
