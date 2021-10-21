package io.kinference

import io.kinference.utils.KILogger
import io.kinference.utils.LoggerFactory

actual object TestLoggerFactory {
    actual fun create(name: String): KILogger = LoggerFactory.create(name)
}
