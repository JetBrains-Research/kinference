package io.kinference.utils.time

import kotlin.js.Date

actual object Time {
    internal actual fun epochMillis(): Long {
        return Date.now().toLong()
    }
}
