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
    container("Build", "amazoncorretto:17") {
        addAwsKeys()

        cache {
            storeKey = "test-data-{{ hashFiles('buildSrc/src/main/kotlin/io/kinference/gradle/s3/DefaultS3Deps.kt') }}"
            localPath = "test-data/*"
        }

        cache {
            storeKey = "maven-{{ hashFiles('**/*gradle.kts') }}"
            localPath = "/root/.m2/repository/*"
        }


        kotlinScript { api ->
            api.gradlew("build", "-Pci", "-Pdisable-tests", "--console=plain")
        }
    }
}

fun Container.addAwsKeys() {
    env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
    env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")
}

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
