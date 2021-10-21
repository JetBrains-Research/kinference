package io.kinference.utils

import org.slf4j.Logger
import org.slf4j.LoggerFactory

class Slf4jLogger(private val logger: Logger): KILogger {
    override fun debug(message: () -> String) {
        logger.debug(message())
    }

    override fun info(message: () -> String) {
        logger.info(message())
    }

    override fun warning(message: () -> String) {
        logger.warn(message())
    }

    override fun error(message: () -> String, e: Throwable?) {
        logger.error(message(), e)
    }

}


actual object LoggerFactory {
    actual fun create(name: String): KILogger {
        val logger = LoggerFactory.getLogger(name)
        return Slf4jLogger(logger)
    }
}
