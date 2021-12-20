package io.kinference.utils.webgpu

actual enum class BufferBindingType(val value: String) {
    Storage("storage"),
    ReadOnlyStorage("read-only-storage"),
}
typealias GPUBufferBindingType = String

actual enum class BufferUsage(val value: Long) {
    MapRead(0x0001),
    MapWrite(0x0002),
    CopySrc(0x0004),
    CopyDst(0x0008),
    Index(0x0010),
    Storage(0x0080),
    Indirect(0x0100),
    QueryResolve(0x0200),
}

actual enum class CompilationMessageType(val value: String) {
    Error("error"),
    Warning("warning"),
    Info("info"),
}
typealias GPUCompilationMessageType = String

actual enum class MapMode(val value: Long) {
    Read(0x0001),
    Write(0x0002),
}

actual enum class PowerPreference(val value: String) {
    LowPower("low-power"),
    HighPerformance("high-performance"),
}
typealias GPUPowerPreference = String
