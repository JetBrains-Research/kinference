FROM ubuntu:22.04

ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y wget xvfb git gnupg software-properties-common

RUN add-apt-repository ppa:mozillateam/ppa
RUN printf "\
Package: *\n\
Pin: release o=LP-PPA-mozillateam\n\
Pin-Priority: 1001\
" | tee /etc/apt/preferences.d/mozilla-firefox

RUN apt-get install -y firefox

#Install Amazon Corretto JDK 17
RUN wget -q -O - https://apt.corretto.aws/corretto.key | apt-key add - \
    && echo "deb https://apt.corretto.aws stable main" >> /etc/apt/sources.list.d/amazoncoretto.list

RUN apt-get update && \
    apt-get install -y java-17-amazon-corretto-jdk
