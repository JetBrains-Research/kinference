package io.kinference.utils.webgpu

expect enum class BufferBindingType {
    Storage,
    ReadOnlyStorage,
}

expect enum class BufferUsage {
    MapRead,
    MapWrite,
    CopySrc,
    CopyDst,
    Index,
    Storage,
    Indirect,
    QueryResolve,
}

expect enum class CompilationMessageType {
    Error,
    Warning,
    Info,
}

expect enum class MapMode {
    Read,
    Write,
}

expect enum class PowerPreference {
    LowPower,
    HighPerformance
}
