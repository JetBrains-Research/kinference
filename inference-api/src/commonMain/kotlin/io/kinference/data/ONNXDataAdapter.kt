package io.kinference.data

interface ONNXDataAdapter<SourceType, TargetType : ONNXData<*>> {
    fun toONNXData(data: SourceType): TargetType
    fun fromONNXData(data: TargetType): SourceType
}

object IdAdapter : ONNXDataAdapter<ONNXData<*>, ONNXData<*>>  {
    override fun toONNXData(data: ONNXData<*>): ONNXData<*> = data
    override fun fromONNXData(data: ONNXData<*>): ONNXData<*> = data
}
