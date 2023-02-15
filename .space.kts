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

job("KInference / Build and Test") {
    container("Build with Gradle", "amazoncorretto:17") {
        addAwsKeys()

//        mountDir = "/root"

//        cache {
//            storeKey = "test-data-{{ hashFiles('buildSrc/src/main/kotlin/io/kinference/gradle/s3/DefaultS3Deps.kt') }}"
//            localPath = "test-data/*"
//        }

//        cache {
//            storeKey = "maven-{{ hashFiles('**/*gradle.kts') }}"
//            localPath = "~/.m2/repository"
//        }
//
//        cache {
//            storeKey = "node_modules-{{ hashFiles('kotlin-js-store/yarn.lock') }}"
//            localPath = "build/js/node_modules"
//        }


//        shellScript("Build with Gradle") {
//            content = """
//                ./gradlew assemble --parallel --console=plain --no-daemon
//                find / -type d -name ".m2"
//                $packBuildFolders
//                """.trimIndent()
//
//        }
        shellScript {
            content = """
                shopt -s extglob
                test="`find !(serialization) -type d -name 'commonMain'` serialization"
                echo ${'$'}test
            """.trimIndent()
        }

//        kotlinScript("Build with Gradle") { api ->
//            api.gradlew("assemble", "--parallel", "--console=plain")
//
//        }
    }
}

fun Container.addAwsKeys() {
    env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
    env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")
}

val packBuildFolders = """
    shopt -s extglob
    build_folders="`find !(build) -type d -name 'build'` build"
    for folder in ${'$'}build_folders
    do
        mkdir -p ${'$'}JB_SPACE_FILE_SHARE_PATH/${'$'}folder
        cp -R ${'$'}folder ${'$'}JB_SPACE_FILE_SHARE_PATH/${'$'}folder
    done
""".trimIndent()

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
