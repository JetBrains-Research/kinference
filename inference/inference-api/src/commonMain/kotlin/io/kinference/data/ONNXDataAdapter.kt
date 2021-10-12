package io.kinference.data

interface ONNXDataAdapter<SourceType> {
    fun toONNXData(name: String, data: SourceType): ONNXData<*>
    fun fromONNXData(data: ONNXData<*>): SourceType
}

object IdAdapter : ONNXDataAdapter<ONNXData<*>>  {
    override fun toONNXData(name: String, data: ONNXData<*>): ONNXData<*> = data
    override fun fromONNXData(data: ONNXData<*>): ONNXData<*> = data
}
