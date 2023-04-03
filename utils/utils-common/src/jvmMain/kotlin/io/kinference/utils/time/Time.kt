package io.kinference.utils.time

actual object Time {
    internal actual fun epochMillis(): Long {
        return System.currentTimeMillis()
    }
}
