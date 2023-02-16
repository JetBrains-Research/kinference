/*job("KInference / Build and Test") {
    host("Build and test") {
        env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
        env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")

        shellScript("Install Firefox, xvfb and JDK") {
            content = """
                apt-get update && apt-get install firefox xvfb openjdk-17-jdk -y -f
            """.trimIndent()
        }

        kotlinScript("Build with Gradle") { api ->
            api.gradlew("build", "-Pci", "-Pdisable-tests", "--console=plain")
        }

        shellScript("Run tests") {
            content = """
                xvfb-run --auto-servernum ./gradlew -Pci jvmTest jsLegacyTest jsIrTest jsTest --console=plain
            """.trimIndent()
        }

        shellScript("Run heavy tests") {
            content = """
                xvfb-run --auto-servernum ./gradlew -Pci jvmHeavyTest jsLegacyHeavyTest jsIrHeavyTest --console=plain
            """.trimIndent()
        }
    }
}*/

val jsContainer = "registry.jetbrains.team/p/ki/containers-ci/ci-corretto-17-firefox:1.0.0"
val jvmContainer = "amazoncorretto:17"

job("KInference / Build and Test") {
    parallel {
        container("Build With Gradle",jvmContainer) {
            kotlinScript { api ->
                api.gradlew("assemble", "--parallel", "--console=plain")
            }
        }

        container("JVM Tests", jvmContainer) {
            kotlinScript { api ->
                api.gradlew("jvmTest", "--parallel", "--console=plain", "-Pci")
            }
        }

        container("JS Legacy Tests", jsContainer) {
            shellScript {
                content = "xvfb-run --auto-servernum ./gradlew jsLegacyTest --parallel --console=plain -Pci"
            }
        }

        container("JS IR Tests", jsContainer) {
            shellScript {
                content = "xvfb-run --auto-servernum ./gradlew jsIrTest jsTest --parallel --console=plain -Pci"
            }
        }

        container("JVM Heavy Tests", jvmContainer) {
            addAwsKeys()
            cacheTestData()

            kotlinScript { api ->
                api.gradlew("jvmHeavyTest", "--console=plain", "-Pci")
            }
        }

        container("JS Legacy Heavy Tests", jsContainer) {
            addAwsKeys()
            cacheTestData()

            shellScript {
                content = "xvfb-run --auto-servernum ./gradlew jsLegacyHeavyTest --console=plain -Pci"
            }
        }

        container("JS IR Heavy Tests", jsContainer) {
            addAwsKeys()
            cacheTestData()

            shellScript {
                content = "xvfb-run --auto-servernum ./gradlew jsIrHeavyTest --console=plain -Pci"
            }
        }
    }
}

fun Container.addAwsKeys() {
    env["AWS_ACCESS_KEY"] = "{{ project:aws_access_key }}" //Secrets("aws_access_key")
    env["AWS_SECRET_KEY"] = "{{ project:aws_secret_key }}" //Secrets("aws_secret_key")
}

fun Container.cacheTestData() {
    cache {
        storeKey = "test-data-{{ hashFiles('buildSrc/src/main/kotlin/io/kinference/gradle/s3/DefaultS3Deps.kt') }}"
        localPath = "test-data"
    }
}

/*val packBuildFolders = """
    shopt -s extglob
    build_folders="`find !(build) -type d -name 'build'` build"
    for folder in ${'$'}build_folders
    do
        mkdir -p ${'$'}JB_SPACE_FILE_SHARE_PATH/${'$'}folder
        cp -R ${'$'}folder ${'$'}JB_SPACE_FILE_SHARE_PATH/${'$'}folder
    done
""".trimIndent()*/

job("KInference / Release") {
    startOn {
        gitPush {
            enabled = false
        }
    }

    container("amazoncorretto:17") {
        env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
        env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")


        shellScript("Release") {
            content = """
                ./gradlew publish
            """.trimIndent()
        }
    }
}
