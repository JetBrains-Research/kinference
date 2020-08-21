package org.jetbrains.research.kotlin.inference

import org.jetbrains.kotlin.cli.common.CLIConfigurationKeys
import org.jetbrains.kotlin.cli.common.environment.setIdeaIoUseFallback
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.cli.jvm.compiler.EnvironmentConfigFiles
import org.jetbrains.kotlin.cli.jvm.compiler.KotlinCoreEnvironment
import org.jetbrains.kotlin.cli.jvm.config.addJvmClasspathRoots
import org.jetbrains.kotlin.com.intellij.openapi.Disposable
import org.jetbrains.kotlin.config.CommonConfigurationKeys
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.metadata.jvm.deserialization.JvmProtoBufUtil
import org.jetbrains.kotlin.utils.PathUtil
import java.io.File

object EnvironmentManager {
    /** Create KotlinCoreEnvironment with specified classpath */
    fun create(libraries: Set<File>): KotlinCoreEnvironment {
        setIdeaIoUseFallback()

        val configuration = CompilerConfiguration().apply {
            addJvmClasspathRoots(PathUtil.getJdkClassesRootsFromCurrentJre() + libraries)
            put(CommonConfigurationKeys.MODULE_NAME, JvmProtoBufUtil.DEFAULT_MODULE_NAME)
            put(CLIConfigurationKeys.MESSAGE_COLLECTOR_KEY, MessageCollector.NONE)
        }

        return KotlinCoreEnvironment.createForProduction(Disposable { }, configuration, EnvironmentConfigFiles.JVM_CONFIG_FILES)
    }
}
