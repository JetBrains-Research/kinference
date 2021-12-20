package io.kinference.webgpu.utils

infix fun Int.divUp(divisor: Int) = (this + divisor - 1) / divisor
