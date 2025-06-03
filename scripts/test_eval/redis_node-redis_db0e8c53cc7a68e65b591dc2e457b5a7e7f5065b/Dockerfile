FROM node:6

RUN sed -i 's|http://deb.debian.org/debian stretch|http://archive.debian.org/debian stretch|g; s|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g; /stretch-updates/d' /etc/apt/sources.list

RUN apt-get update && apt-get install -y redis-server

COPY test.sh /test.sh

WORKDIR /repo

CMD ["bash", "-c", "/test.sh"]