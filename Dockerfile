FROM ubuntu:latest
MAINTAINER cadot.info "docker@cadot.info"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN apt-get install unzip wget libjpeg-dev libpng-dev \
 libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
 libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran cmake python3-pip nano -y
RUN pip install numpy

RUN wget https://github.com/opencv/opencv/archive/master.zip
RUN unzip master.zip
RUN rm master.zip
RUN wget https://github.com/opencv/opencv_contrib/archive/master.zip
RUN unzip master.zip
RUN rm master.zip
RUN wget https://bootstrap.pypa.io/get-pip.py


WORKDIR "/opencv-master"
RUN mkdir build
WORKDIR "/opencv-master/build"
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib-master/modules \
	-D PYTHON_EXECUTABLE=/bin/python3 \
	-D BUILD_EXAMPLES=ON ..
RUN make -j4
RUN make install
RUN ldconfig
RUN pip install matplotlib
RUN pip install rembg

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
		curl \
		apache2 \
		software-properties-common \
		sqlite3 \
		supervisor \
		dirmngr \
                gpg-agent \
	&& apt-get clean \
	&& rm -fr /var/lib/apt/lists/*

RUN LC_ALL=C.UTF-8 add-apt-repository ppa:ondrej/php
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list

# pour mongodb RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 68818C72E52529D4 

#RUN echo "deb [ arch=amd64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.0.list

RUN apt-get update && apt-get install -y --no-install-recommends \
	yarn \	
	libapache2-mod-php7.4 \
		nodejs \
		npm \
		php7.4 \
                php7.4-intl \
		php7.4-cli \
		php7.4-curl \
		php7.4-dev \
		php7.4-gd \
		php7.4-imap \
		php7.4-mbstring \
		php-xml \ 
		#php7.4-mcrypt \
		php7.4-mysql \
		php7.4-pgsql \
		php7.4-pspell \
		php7.4-xml \
		php7.4-sqlite3 \
		php7.4-xmlrpc \
		php7.4-zip \
                php7.4-soap \
		php-apcu \
		php-memcached \
		php-pear \
		php-redis \
		php7.4-xdebug \
		php7.4-xml\
		nano \
		php-swiftmailer \
		git \
                wkhtmltopdf \
		unzip \
		phpunit \
                build-essential \
		curl \
		wget \
		jpegoptim \
		optipng \
		pngquant \
		gifsicle \
		webp \
		php7.4-tidy \
		php7.4-bcmath \
		#mongodb-org \
	&& apt-get clean \
	&& rm -fr /var/lib/apt/lists/*


RUN apt-get install -f
RUN pip install random2
RUN a2enmod rewrite
RUN a2enmod expires
COPY /000-default.conf /etc/apache2/sites-available/000-default.conf

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY php.ini /etc/php/7.4/apache2/php.ini
COPY php.ini /etc/php/7.4/cli/php.ini
COPY restart.sh /var/www/html/.
EXPOSE 8080
RUN cd /
WORKDIR "/var/www/html"
RUN pip install imutils
RUN pip install flask_cors
RUN pip install uuid
RUN /usr/bin/python3 -m pip install --upgrade pip
COPY server.py /.
CMD [ "python", "/server.py" ]
