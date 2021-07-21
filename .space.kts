job("KInference / Build") {
    container("openjdk:11") {
        env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
        env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")

        shellScript {
            content = """
              ./gradlew build  
          """
        }
    }
}

job("KInference / JVM / Test") {
    container("openjdk:11") {
        env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
        env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")

        shellScript {
            content = """
              ./gradlew jvmTest  
          """
        }
    }
}

job("KInference / JVM / Heavy Test") {
    container("openjdk:11") {
        env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
        env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")

        shellScript {
            content = """
              ./gradlew jvmHeavyTest  
          """
        }
    }
}

job("KInference / Release") {
    startOn {
        gitPush {
            enabled = false
        }
    }

    container("openjdk:11") {
        env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
        env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")

        shellScript {
            content = """
              ./gradlew publish    
          """
        }
    }
}
