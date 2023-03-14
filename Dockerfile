FROM alpine

WORKDIR /app
COPY . .

ARG DATASET_AUTHOR
ARG DATASET_NAME
ARG DATASET_PATH

ARG KAGGLE_USERNAME
ARG KAGGLE_KEY

ENV KAGGLE_USERNAME=${KAGGLE_USERNAME}
ENV KAGGLE_KEY=${KAGGLE_KEY}

RUN apk add --no-cache python3 \
    && apk add --no-cache py3-pip \
    && pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt \
    && env \
    && kaggle datasets download -d ${DATASET_AUTHOR}/${DATASET_NAME} -p ${DATASET_PATH} \
    && unzip ${DATASET_PATH}/${DATASET_NAME}.zip -d ${DATASET_PATH}/artworks \
    && rm ${DATASET_PATH}/${DATASET_NAME}.zip \
    && apk del py-pip \
    && apk --purge del apk-tools

CMD ["yes"]